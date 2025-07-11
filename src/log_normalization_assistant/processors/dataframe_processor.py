import pandas as pd
from typing import List, Optional, Dict, Any


class DataFrameProcessor:
    """
    Processes a DataFrame to build an index DataFrame based on its structural classification.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        structure_class: str,
        analysis_result: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Initialize the DataFrameProcessor with the DataFrame and its classification.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to process for indexing.
        structure_class : str
            The structural classification label (e.g., 'perfect_structure',
            'one_splitter', 'multiple_splitters').
        analysis_result : list of dict, optional
            The detailed analysis results for partition boundaries when handling
            multiple splitters.
        """
        self.df = df
        self.total_rows = df.shape[0]
        self.structure_class = structure_class
        self.analysis_result = analysis_result

    def create_index_df(self) -> pd.DataFrame:
        """
        Create an index DataFrame based on the structure classification.

        Determines which partitioning method to apply and returns a DataFrame with
        columns 'main_part', 'table_start_idx', and 'table_end_idx' marking each
        partition's indices.

        Returns
        -------
        pandas.DataFrame
            The index DataFrame mapping structure partitions.
        """
        if self.structure_class == 'perfect_structure':
            return self._create_perfect_structure_df()
        elif self.structure_class == 'one_splitter':
            return self._create_one_splitter_df()
        elif self.structure_class == 'multiple_splitters':
            return self._create_multiple_splitters_df()
        return self._create_perfect_structure_df()

    def _create_perfect_structure_df(self) -> pd.DataFrame:
        """
        Generate an index for DataFrames with a perfect single table layout.

        Returns
        -------
        pandas.DataFrame
            A single-row DataFrame where 'table_start_idx' is the second row index
            and 'table_end_idx' is the last row index.
        """
        return pd.DataFrame({
            'main_part': [[self.df.index[0]]],
            'table_start_idx': [self.df.index[1]],
            'table_end_idx': [self.df.index[-1]]
        })

    def _create_one_splitter_df(self) -> pd.DataFrame:
        """
        Generate indices for DataFrames with exactly one splitter marker.

        Identifies rows where 'Unnamed: 0' is non-null as splitters and computes
        start and end indices for each resulting table partition.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per splitter containing:
            - 'main_part': the splitter row index
            - 'table_start_idx': the first data row after the splitter
            - 'table_end_idx': the last data row before the next splitter or end
        """
        split_idxs = self.df.index[~self.df['Unnamed: 0'].isna()].tolist()
        starts = [
            self.df.index[self.df.index.get_loc(i) + 1]
            for i in split_idxs
            if self.df.index.get_loc(i) + 1 < self.total_rows
        ]
        ends = [
            self.df.index[self.df.index.get_loc(j) - 1]
            for j in split_idxs[1:]
        ] + [self.df.index[-1]]

        return pd.DataFrame([
            {
                'main_part': [split_idx],
                'table_start_idx': start_idx,
                'table_end_idx': end_idx
            }
            for split_idx, start_idx, end_idx in zip(split_idxs, starts, ends)
            if start_idx is not None and end_idx is not None
        ])

    def _create_multiple_splitters_df(self) -> pd.DataFrame:
        """
        Generate indices for DataFrames with multiple splitter markers using AI analysis.

        Uses 'analysis_result' containing AI-determined boundaries for each partition.
        Applies row-prefix formatting to each index value.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per analyzed partition containing:
            - 'main_part': list of original splitter indices (prefixed)
            - 'table_start_idx': prefixed start index for the table
            - 'table_end_idx': prefixed end index for the table
        """
        if not self.analysis_result:
            return self._create_perfect_structure_df()

        rows: List[Dict[str, Any]] = []
        for item in self.analysis_result:
            table = item.get('table', {})
            rows.append({
                'main_part': item.get('main_part', []),
                'table_start_idx': table.get('start'),
                'table_end_idx': table.get('end'),
            })

        if not rows:
            return self._create_perfect_structure_df()

        df_idx = pd.DataFrame(rows)

        def prefix_rows(x: Any) -> Any:
            """
            Prefix row indices or lists of indices with 'row'.

            Parameters
            ----------
            x : Any
                A scalar index or list of indices.

            Returns
            -------
            Any
                Prefixed index string or list of strings.
            """
            if x is None:
                return None
            if isinstance(x, list):
                return [f"row{v}" for v in x]
            return f"row{x}"

        for col in ['main_part', 'table_start_idx', 'table_end_idx']:
            df_idx[col] = df_idx[col].apply(prefix_rows)

        return df_idx

