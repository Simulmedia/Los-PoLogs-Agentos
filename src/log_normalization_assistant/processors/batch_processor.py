from typing import List

import pandas as pd


class BatchProcessor:
    """
    Splits index DataFrames into smaller batches for incremental processing.
    """

    def __init__(self, batch_size: int = 30) -> None:
        """
        Initialize the BatchProcessor with a specified batch size.

        Parameters
        ----------
        batch_size : int, optional
            The number of rows per batch (default is 30).
        """
        self.batch_size = batch_size

    def process_batches(
        self,
        ind_df: pd.DataFrame,
        structure_class: str,
        original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Split each index row into multiple batch rows based on batch size.

        Iterates through each partition defined in ind_df and divides the
        range between 'table_start_idx' and 'table_end_idx' into chunks of
        size batch_size, returning a concatenated DataFrame of all batches.

        Parameters
        ----------
        ind_df : pandas.DataFrame
            Index DataFrame with 'table_start_idx' and 'table_end_idx' for partitions.
        structure_class : str
            The structural classification label guiding main_part handling.
        original_df : pandas.DataFrame
            The original DataFrame used to map index locations to labels.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with rows for each batch containing updated start and end indices.
        """
        if ind_df.empty:
            return ind_df

        all_batches: List[pd.DataFrame] = []
        for _, row in ind_df.iterrows():
            all_batches.append(
                self._split_row_into_batches(row, structure_class, original_df)
            )

        return pd.concat(all_batches, ignore_index=True)

    def _split_row_into_batches(
        self,
        row: pd.Series,
        structure_class: str,
        original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Split a single partition row into batch-sized segments.

        Calculates start and end locations in the original DataFrame and
        generates a set of batch rows, adjusting 'main_part' for perfect structure.

        Parameters
        ----------
        row : pandas.Series
            A row from the index DataFrame containing 'table_start_idx' and 'table_end_idx'.
        structure_class : str
            The structural classification label guiding main_part handling.
        original_df : pandas.DataFrame
            The original DataFrame used to resolve index positions.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing one or more batch rows derived from the input row.
        """
        start_idx = row.table_start_idx
        end_idx = row.table_end_idx
        batches: List[pd.Series] = []

        if pd.isna(start_idx) or pd.isna(end_idx):
            return pd.DataFrame([row])

        start_loc = original_df.index.get_loc(start_idx)
        end_loc = original_df.index.get_loc(end_idx)

        for i in range(start_loc, end_loc + 1, self.batch_size):
            batch_end_loc = min(i + self.batch_size - 1, end_loc)
            new_row = row.copy()
            new_row.table_start_idx = original_df.index[i]
            new_row.table_end_idx = original_df.index[batch_end_loc]

            if structure_class == 'perfect_structure':
                new_row.main_part = [original_df.index[i]]
                if i + 1 <= end_loc:
                    new_row.table_start_idx = original_df.index[i + 1]

            batches.append(new_row)
        return pd.DataFrame(batches)