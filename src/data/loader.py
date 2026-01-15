"""
Data loading utilities for Excel and Parquet files.

Handles loading of strategy data files, folder-based batch loading, and format conversion.
Uses joblib for parallel loading to improve performance on large datasets.
"""
# pylint: disable=broad-exception-caught,too-many-locals

import logging
import os
import re
import warnings
from typing import List, Optional, Tuple

import pandas as pd

# Parallel loading support
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_n_jobs() -> int:
    """Get optimal number of parallel jobs."""
    try:
        cpu_count = os.cpu_count() or 4
        # Use half of CPUs, minimum 2, maximum 8
        return min(max(2, cpu_count // 2), 8)
    except Exception:
        return 4


class DataLoader:
    """Static utility class for loading and processing trading data files."""

    @staticmethod
    def load_file(filepath: str) -> Optional[pd.DataFrame]:
        """
        Load a single data file (Excel or Parquet).

        Excel structure:
        - Row 0: Date at col 0, 10, 20... (block headers)
        - Row 1: Column headers (No., Profit, Trades, PF, DD %, etc.)
        - Row 2+: Data

        Args:
            filepath: Path to the file

        Returns:
            DataFrame with loaded data or None if failed
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # Parquet files - direct load
            if filepath.endswith(".parquet"):
                logger.info("Loading parquet: %s", filepath)
                return pd.read_parquet(filepath)

            # Excel files - parse block structure
            logger.info("Loading Excel: %s", filepath)
            df_raw = pd.read_excel(filepath, header=None)

            all_blocks = []

            # Forward fill the first row (Dates) to handle merged cells
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                row0_filled = df_raw.iloc[0].ffill().infer_objects(copy=False)

            row1 = df_raw.iloc[1]

            # Find block starts (columns with "No." in row 1)
            block_starts = [i for i, val in enumerate(row1) if str(val).strip() == "No."]

            for start_col in block_starts:
                date_val = row0_filled[start_col]

                # Expected headers for each block
                headers = df_raw.iloc[1, start_col : start_col + 10].tolist()

                col_map = {}
                target_cols = [
                    "No.",
                    "Profit",
                    "Trades",
                    "PF",
                    "DD %",
                    "Start",
                    "1st Candle",
                    "Shift",
                    "Position",
                ]

                for i, h in enumerate(headers):
                    h_str = str(h).strip()
                    if h_str in target_cols and h_str not in col_map:
                        col_map[h_str] = start_col + i

                if "No." not in col_map or "Profit" not in col_map:
                    continue

                # Extract data for found columns
                block_data = pd.DataFrame()
                for col_name, col_idx in col_map.items():
                    block_data[col_name] = df_raw.iloc[2:, col_idx]

                block_data["Date"] = date_val

                # Drop rows where 'No.' is NaN
                if "No." in block_data.columns:
                    block_data = block_data.dropna(subset=["No."])

                all_blocks.append(block_data)

            if all_blocks:
                final_df = pd.concat(all_blocks, ignore_index=True)
                logger.info(
                    "Loaded %s: %d rows, %d blocks",
                    os.path.basename(filepath),
                    len(final_df),
                    len(all_blocks),
                )
                return final_df

            logger.warning("No data blocks found in %s", filepath)
            return None

        except Exception as e:
            logger.error("Error loading %s: %s", filepath, e)
            raise RuntimeError(f"Error loading {filepath}: {e}") from e

    @staticmethod
    def _load_single_file(folder_path: str, filename: str) -> Optional[pd.DataFrame]:
        """Load a single file with error handling (for parallel loading)."""
        path = os.path.join(folder_path, filename)
        try:
            df = DataLoader.load_file(path)
            if df is not None:
                if "SourceFile" not in df.columns:
                    df["SourceFile"] = filename
                return df
        except Exception as e:
            logger.error("Failed to load %s: %s", filename, e)
        return None

    @staticmethod
    def load_folder(folder_path: str, n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        Load all compatible files from a folder using parallel processing.

        Args:
            folder_path: Path to folder containing data files
            n_jobs: Number of parallel jobs (default: auto)

        Returns:
            Combined DataFrame with all data
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find all Excel and Parquet files
        files = [
            f
            for f in os.listdir(folder_path)
            if f.endswith(".xlsx") or f.endswith(".parquet")
        ]

        if not files:
            logger.warning("No Excel or Parquet files found in %s", folder_path)
            return pd.DataFrame()

        logger.info("Loading %d files from %s", len(files), folder_path)

        # Determine number of jobs
        if n_jobs is None:
            n_jobs = _get_n_jobs()

        # Use parallel loading if joblib available and multiple files
        if JOBLIB_AVAILABLE and len(files) > 1:
            logger.info("Using parallel loading with %d workers", n_jobs)
            try:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(DataLoader._load_single_file)(folder_path, f)
                    for f in files
                )
                all_data = [r for r in results if r is not None]
            except Exception as e:
                logger.warning("Parallel loading failed (%s), falling back to sequential", e)
                all_data = []
                for filename in files:
                    df = DataLoader._load_single_file(folder_path, filename)
                    if df is not None:
                        all_data.append(df)
        else:
            # Sequential loading
            all_data = []
            for filename in files:
                df = DataLoader._load_single_file(folder_path, filename)
                if df is not None:
                    all_data.append(df)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info("Total loaded: %d rows from %d files", len(result), len(all_data))
            return result
        return pd.DataFrame()

    @staticmethod
    def load_parquet_files(file_paths: List[str]) -> pd.DataFrame:
        """
        Load multiple parquet files.

        Args:
            file_paths: List of file paths

        Returns:
            Combined DataFrame
        """
        all_data = []
        for path in file_paths:
            try:
                df = pd.read_parquet(path)
                if "SourceFile" not in df.columns:
                    df["SourceFile"] = os.path.basename(path)
                all_data.append(df)
                logger.info("Loaded parquet: %s (%d rows)", os.path.basename(path), len(df))
            except Exception as e:
                logger.error("Failed to load %s: %s", path, e)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    @staticmethod
    def convert_excel_to_parquet(
        source_folder: str, output_folder: Optional[str] = None, n_jobs: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Convert all Excel files in a folder to a single Parquet file.
        Uses parallel loading for better performance.

        The output filename is generated based on the input filenames (years).
        Example: AUDJPY_2020_Weekly_Fix.xlsx, AUDJPY_2021_Weekly_Fix.xlsx
                 -> AUDJPY_2020_2021_Weekly_Fix.parquet

        Args:
            source_folder: Folder containing Excel files
            output_folder: Output folder (defaults to source_folder)
            n_jobs: Number of parallel jobs (default: auto)

        Returns:
            Tuple of (output_path, row_count)
        """
        if output_folder is None:
            output_folder = source_folder

        # Find all Excel files
        files = [f for f in os.listdir(source_folder) if f.endswith(".xlsx")]
        if not files:
            raise FileNotFoundError("No Excel files found in the selected folder.")

        logger.info("Converting %d Excel files to Parquet", len(files))

        # Determine number of jobs
        if n_jobs is None:
            n_jobs = _get_n_jobs()

        # Load all files (parallel if available)
        if JOBLIB_AVAILABLE and len(files) > 1:
            logger.info("Using parallel loading with %d workers", n_jobs)
            try:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(DataLoader._load_single_file)(source_folder, f)
                    for f in files
                )
                all_data = [r for r in results if r is not None]
            except Exception as e:
                logger.warning("Parallel loading failed (%s), falling back to sequential", e)
                all_data = []
                for filename in files:
                    df = DataLoader._load_single_file(source_folder, filename)
                    if df is not None:
                        all_data.append(df)
        else:
            all_data = []
            for filename in files:
                df = DataLoader._load_single_file(source_folder, filename)
                if df is not None:
                    all_data.append(df)

        if not all_data:
            raise ValueError("Failed to load any data from Excel files.")

        merged_df = pd.concat(all_data, ignore_index=True)

        # Generate output filename from years
        years = set()
        for f in files:
            found_years = re.findall(r"\d{4}", f)
            years.update(found_years)

        sorted_years = sorted(list(years))
        year_str = "_".join(sorted_years)

        # Build output filename
        first_file = files[0]
        name_without_ext = os.path.splitext(first_file)[0]

        if sorted_years:
            year_in_first = next((y for y in sorted_years if y in name_without_ext), None)
            if year_in_first:
                output_name = name_without_ext.replace(year_in_first, year_str)
            else:
                output_name = f"{name_without_ext}_{year_str}"
        else:
            output_name = f"{name_without_ext}_merged"

        output_filename = f"{output_name}.parquet"
        output_path = os.path.join(output_folder, output_filename)

        # Handle PF column (empty cells as 0)
        if "PF" in merged_df.columns:
            merged_df["PF"] = pd.to_numeric(merged_df["PF"], errors="coerce").fillna(0)

        # Save to Parquet
        merged_df.to_parquet(output_path, index=False)

        logger.info("Saved parquet: %s (%d rows)", output_path, len(merged_df))
        return output_path, len(merged_df)

    @staticmethod
    def get_file_list(folder_path: str) -> List[str]:
        """
        Get list of data files in a folder.

        Args:
            folder_path: Path to folder

        Returns:
            List of filenames
        """
        if not os.path.exists(folder_path):
            return []

        return [
            f
            for f in os.listdir(folder_path)
            if f.endswith(".xlsx") or f.endswith(".parquet") or f.endswith(".csv")
        ]
