"""
Module for analyzing and classifying pandas DataFrame structures using an AI-based approach.

This module defines the FileStructureAnalyzer class, which connects to the OpenAI API
for structure classification and partition analysis, with support for multithreading.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from typing import Any, Dict, List

from config.settings import SYSTEM_PROMPT1_PATH, SYSTEM_PROMPT2_PATH, PRIMARY_MODEL
from src.log_normalization_assistant.utils import load_text_file


class FileStructureAnalyzer:
    """
    Analyzes the structure of pandas DataFrames to classify their layout and extract
    meaningful partitions by interacting with an AI model. Supports single-threaded
    and multithreaded analysis for efficient processing of large partition sets.
    """

    def __init__(self, api_key: str, model: str = PRIMARY_MODEL, max_workers: int = 16, temperature: float = 0.1) -> None:
        """
        Initialize the FileStructureAnalyzer with API credentials and model selection.

        Parameters
        ----------
        api_key : str
            The OpenAI API key for authentication.
        model : str, optional
            The name of the OpenAI model to use (default is "gpt-4.1-mini").
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.temperature = temperature

    def _get_ai_response(self, system_prompt: str, user_content: str) -> Dict[str, Any]:
        """
        Send a prompt to the AI system and parse the JSON response.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction text.
        user_content : str
            The JSON-serialized sample of DataFrame content for classification.

        Returns
        -------
        Dict[str, Any]
            The parsed JSON response from the AI API. Returns an empty dict on parse error.
        """
        # Prepare messages for ChatCompletion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Use lower temperature for deterministic output with default model
        kwargs = {"model": self.model, "messages": messages}
        if self.model == PRIMARY_MODEL:
            kwargs["temperature"] = self.temperature

        # Request completion
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        # Strip markdown fences if present
        json_text = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.DOTALL)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from AI response:\n{content}")
            return {}

    def classify_structure(self, df: pd.DataFrame) -> str:
        """
        Classify the structure of a DataFrame by sampling its first rows and columns.

        This method takes a sample of the DataFrame (first five rows and eight columns),
        serializes it to JSON, and sends it to the AI system with a defined system prompt
        for structural classification.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame whose structure is to be classified.

        Returns
        -------
        str
            The structure class label returned by the AI system, or 'unknown_structure'
            if classification fails or no class is returned.
        """
        nested_data = df.iloc[:5, :8].to_dict('index')
        user_content = json.dumps(nested_data, default=str)

        system_prompt = load_text_file(SYSTEM_PROMPT1_PATH)
        response = self._get_ai_response(system_prompt, user_content)
        return response.get('class', 'unknown_structure')

    def _analyze_single_splitter_part(
        self,
        df_part: pd.DataFrame,
        system_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze a single partition of the DataFrame for detailed structure information.

        Designed for concurrent execution, this helper method serializes the partition
        to JSON and queries the AI system using the provided prompt.

        Parameters
        ----------
        df_part : pandas.DataFrame
            A slice of the original DataFrame corresponding to one main splitter section.
        system_prompt : str
            The system prompt text for partition analysis.

        Returns
        -------
        Dict[str, Any]
            The AI-provided analysis of the given DataFrame partition.
        """
        nested_data_json = df_part.to_dict('index')
        user_content_json = json.dumps(nested_data_json, default=str)
        return self._get_ai_response(system_prompt, user_content_json)

    def analyze_multiple_splitters(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze DataFrames containing multiple main splitters to identify sub-partitions.

        Detects non-null splitter markers in the 'Unnamed: 0' index column to delineate
        sections, then applies single-part analysis concurrently when multiple sections
        are detected, or sequentially otherwise.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame potentially containing multiple splitter sections.

        Returns
        -------
        List[Dict[str, Any]]
            A list of analysis results for each detected section.
        """
        main_splitter_indices = df.index[~df['Unnamed: 0'].isna()].tolist()
        system_prompt = load_text_file(SYSTEM_PROMPT2_PATH)

        all_results: List[Dict[str, Any]] = []
        n_splitters = len(main_splitter_indices)

        if n_splitters > 1:
            num_workers = min(n_splitters, self.max_workers)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for i, start_idx in enumerate(main_splitter_indices):
                    end_idx = (main_splitter_indices[i + 1]
                               if i + 1 < n_splitters else df.index[-1])
                    start_loc = df.index.get_loc(start_idx)
                    end_loc = df.index.get_loc(end_idx) - 1 if i + 1 < n_splitters else None
                    df_part = (df.iloc[start_loc:end_loc, :10]
                               if end_loc is not None else df.iloc[start_loc:, :10])
                    futures[executor.submit(
                        self._analyze_single_splitter_part,
                        df_part,
                        system_prompt
                    )] = i

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if isinstance(result, dict):
                            all_results.append(result)
                        elif isinstance(result, list):
                            all_results.extend(result)
                    except Exception as exc:
                        print(f"Splitter {futures[future]} exception: {exc}")
        else:
            df_part = df.iloc[:, :10]
            result = self._analyze_single_splitter_part(df_part, system_prompt)
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, dict):
                all_results.append(result)

        return all_results
