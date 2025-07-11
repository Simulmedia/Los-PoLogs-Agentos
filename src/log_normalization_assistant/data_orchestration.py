import json
from typing import Optional, List, Any, Dict

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from src.log_normalization_assistant import load_text_file, FileStructureAnalyzer, DataFrameProcessor, BatchProcessor
from config.settings import SYSTEM_PROMPT1_PATH, SYSTEM_PROMPT2_PATH, SYSTEM_PROMPT3_PATH, PRIMARY_MODEL, \
    SECONDARY_MODEL


class DataOrchestrator:
    """
    Orchestrates the end-to-end workflow of loading data, analyzing structure,
    processing partitions, and extracting content snippets using AI and multithreading.
    """

    def __init__(self, filepath: str, api_key_path: str, max_workers: int=16, temperature: float=0.1, batch_size: int = 30) -> None:
        """
        Initialize the DataOrchestrator with file paths and load resources.

        Parameters
        ----------
        filepath : str
            Path to the input Excel file to process.
        api_key_path : str
            Path to the file containing the OpenAI API key.
        """
        self.filepath = filepath
        with open(api_key_path, 'r', encoding='utf-8') as f:
            self.api_key = f.read().strip()
        self.client = OpenAI(api_key=self.api_key)

        # Load data and system prompts
        self.df = self._load_data()
        self.SYSTEM_PROMPT1 = load_text_file(SYSTEM_PROMPT1_PATH)
        self.SYSTEM_PROMPT2 = load_text_file(SYSTEM_PROMPT2_PATH)
        self.SYSTEM_PROMPT3 = load_text_file(SYSTEM_PROMPT3_PATH)

        self.max_workers = max_workers
        self.temperature = temperature
        self.batch_size = batch_size

    def _load_data(self) -> pd.DataFrame:
        """
        Load and prepare the Excel file as a pandas DataFrame.

        - Reads the Excel file from self.filepath.
        - Drops rows that are entirely NaN.
        - Renames the index to 'row0', 'row1', etc.

        Returns
        -------
        pandas.DataFrame
            The cleaned DataFrame ready for analysis.
        """
        df = pd.read_excel(self.filepath)
        df.dropna(axis='rows', how='all', inplace=True)
        df.index = [f"row{i}" for i in range(df.shape[0])]
        return df

    def run(self) -> pd.DataFrame:
        """
        Execute the full orchestration pipeline:
        1. Classify DataFrame structure with two AI models.
        2. Generate index DataFrame based on classification.
        3. Split partitions into batches.
        4. Extract and process content snippets concurrently.

        Returns
        -------
        pandas.DataFrame
            The final concatenated DataFrame of processed snippets,
            with numeric columns cleaned and formatted.
        """
        # Step 1: Structure classification
        analyzer1 = FileStructureAnalyzer(api_key=self.api_key, model=PRIMARY_MODEL, max_workers=self.max_workers, temperature=self.temperature)
        analyzer2 = FileStructureAnalyzer(api_key=self.api_key, model=SECONDARY_MODEL, max_workers=self.max_workers, temperature=self.temperature)

        structure_class = analyzer1.classify_structure(self.df)
        print(f"Detected structure class: {structure_class}")

        analysis_result: Optional[List[Dict[str, Any]]] = None

        if structure_class == 'multiple_splitters':
            analysis_result = analyzer2.analyze_multiple_splitters(self.df)

        # Step 2: Create partition index
        processor = DataFrameProcessor(self.df, structure_class, analysis_result)
        ind_df = processor.create_index_df()

        # Step 3: Batch splitting
        batch_processor = BatchProcessor(batch_size=self.batch_size)
        final_df = batch_processor.process_batches(ind_df, structure_class, self.df)

        # Step 4: Snippet extraction
        def process_snippet(row_index: int, row_data: pd.Series) -> pd.DataFrame:
            """
            Extract and process a snippet partition using AI.

            Parameters
            ----------
            row_index : int
                The batch row index (for logging).
            row_data : pandas.Series
                A batch row with 'main_part', 'table_start_idx', and 'table_end_idx'.

            Returns
            -------
            pandas.DataFrame
                The AI-processed snippet data as a DataFrame.
            """
            # Determine rows to include
            main_labels = row_data['main_part']
            idxs = [self.df.index.get_loc(lbl) for lbl in main_labels]
            start_loc = self.df.index.get_loc(row_data['table_start_idx'])
            end_loc = self.df.index.get_loc(row_data['table_end_idx'])
            idxs.extend(range(start_loc, end_loc + 1))

            snippet = self.df.iloc[idxs].drop_duplicates()
            snippet.index = [f"row{j}" for j in range(len(snippet))]
            user_content3 = json.dumps(snippet.to_dict('index'), default=str)

            response = self.client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT3},
                    {"role": "user",   "content": user_content3},
                ],
                temperature=0.1,
            )
            resp_json = json.loads(response.choices[0].message.content)
            data = resp_json if isinstance(resp_json, list) else resp_json.get('data', [])
            return pd.DataFrame(data)

        results: List[pd.DataFrame] = []
        n_snippets = len(final_df)

        num_workers = min(n_snippets, self.max_workers)

        print(f"Spawning up to {num_workers} workers for {n_snippets} snippets.")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_snippet, idx, row): idx
                for idx, row in final_df.iterrows()
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"Error processing snippet {futures[future]}: {exc}")

        # Combine and post-process results
        if results:
            combined = pd.concat(results, ignore_index=True)
            if 'Cost' in combined:
                combined['Cost'] = pd.to_numeric(combined['Cost'], errors='coerce')
                combined['Cost'] = combined['Cost'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
            return combined

        return final_df
