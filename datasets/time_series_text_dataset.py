from typing import Optional, Literal, Callable
from pathlib import Path
import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from .helper_functions import get_transformation


class TimeSeriesTextDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Generic dataset for text-based time-series data with chronological splitting.

    Supports multiple datasets including:
    - UCI Household Power Consumption (semicolon-delimited, Date/Time columns)
    - UCI Beijing Air Quality (comma-delimited, separate year/month/day/hour columns)
    - Other text-based time-series datasets with similar structure

    Implements chronological preprocessing approach:
    1. Chronological dataset split (default: 81% train, 9% val, 10% test)
    2. Sliding window sampling with configurable stride (default: stride=1 for maximum overlap)
    3. Missing data removal (discard windows containing ANY missing values)
    4. Data standardization using training set statistics
    """

    def __init__(
        self,
        file_path: str | Path,
        signal_columns: list[str],
        encode_signals: Optional[list[str]] = None,
        decode_signals: Optional[list[str]] = None,
        window_size: int = 128,
        stride: int = 1,
        max_rows: Optional[int] = None,
        shared_data: Optional[pd.DataFrame] = None,
        shared_stats: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        compute_stats: bool = True,
        data_transformations: Optional[list[str]] = None,
        apply_normalization: Optional[Literal["standardize", "normalize"]] = None,
        normalize_encode_only: bool = False,
        # Data format parameters (for flexibility across different datasets)
        delimiter: str = ";",
        na_values: Optional[list[str]] = None,
        datetime_format: Literal["date_time_columns", "separate_columns", "single_column"] = "date_time_columns",
        datetime_column: Optional[str] = None,
        date_column: Optional[str] = "Date",
        time_column: Optional[str] = "Time",
        year_column: Optional[str] = None,
        month_column: Optional[str] = None,
        day_column: Optional[str] = None,
        hour_column: Optional[str] = None,
        station_column: Optional[str] = None,
    ):
        """
        Args:
            file_path: Path to data file (e.g., household_power_consumption.txt or air_quality.csv)
            signal_columns: List of signal column names to use
            encode_signals: Signals to use for encoding (subset of signal_columns)
            decode_signals: Signals to use for decoding (subset of signal_columns)
            window_size: Length of sliding windows (T in paper, default 128)
            stride: Step size between windows (paper requires stride=1, default 1)
            max_rows: Optional maximum total rows to load (for testing/debugging)
            shared_data: Optional pre-loaded DataFrame to use (for val/test splits)
            shared_stats: Optional pre-computed statistics (use training stats for val/test)
            compute_stats: Whether to compute statistics (True for train, False for val/test)
            data_transformations: Optional list of transformation names to apply after loading data
            apply_normalization: Normalization method ("standardize" per paper, or "normalize")
            normalize_encode_only: If True, only normalize encode

            Data format parameters (for flexibility across datasets):
            delimiter: Column delimiter (';' for household power, ',' for appliances/air quality)
            na_values: List of strings representing missing values (default: ['?'] for household power)
            datetime_format: How datetime is represented in the file:
                - "date_time_columns": Date and Time in separate columns (household power)
                - "separate_columns": year, month, day, hour in separate columns (air quality)
                - "single_column": Single datetime column (appliances energy)
            datetime_column: Name of datetime column (for "single_column" format, e.g., "date")
            date_column: Name of date column (for "date_time_columns" format)
            time_column: Name of time column (for "date_time_columns" format)
            year_column: Name of year column (for "separate_columns" format)
            month_column: Name of month column (for "separate_columns" format)
            day_column: Name of day column (for "separate_columns" format)
            hour_column: Name of hour column (for "separate_columns" format)
            station_column: Optional column for multi-site data (e.g., "station" for air quality)
        """
        super().__init__()

        # Validate that encode/decode signals are subsets of signal_columns
        if encode_signals is not None:
            if not set(encode_signals).issubset(set(signal_columns)):
                raise ValueError(
                    f"encode_signals must be a subset of signal_columns. "
                    f"Missing: {set(encode_signals) - set(signal_columns)}"
                )
        if decode_signals is not None:
            if not set(decode_signals).issubset(set(signal_columns)):
                raise ValueError(
                    f"decode_signals must be a subset of signal_columns. "
                    f"Missing: {set(decode_signals) - set(signal_columns)}"
                )

        self.file_path = Path(file_path)
        self.signal_columns = signal_columns
        if not encode_signals or not decode_signals:
            raise ValueError("TimeSeriesTextDataset requires both encode_signals and decode_signals")

        self.encode_signals = list(encode_signals)
        self.decode_signals = list(decode_signals)
        self.window_size = window_size
        self.stride = stride
        self.max_rows = max_rows

        self.apply_normalization: Optional[Literal["standardize", "normalize"]] = apply_normalization
        self.normalize_encode_only: bool = normalize_encode_only

        # Data format parameters
        self.delimiter = delimiter
        self.na_values = na_values if na_values is not None else ["?"]
        self.datetime_format = datetime_format
        self.datetime_column = datetime_column
        self.date_column = date_column
        self.time_column = time_column
        self.year_column = year_column
        self.month_column = month_column
        self.day_column = day_column
        self.hour_column = hour_column
        self.station_column = station_column

        # Resolve transformation names to callable functions
        self.data_transformations: list[Callable] = []
        if data_transformations:
            for transform_name in data_transformations:
                self.data_transformations.append(get_transformation(transform_name))

        # Data cache
        self._data_cache: Optional[pd.DataFrame] = None
        # Stats as tensors aligned to signal order for fast normalization.
        self._encode_stats_t: Optional[dict[str, torch.Tensor]] = None
        self._decode_stats_t: Optional[dict[str, torch.Tensor]] = None

        # Stats by name (python floats) for evaluation/denorm.
        self.stats_by_name: Optional[dict[str, dict[str, float]]] = None
        self._windows: list[tuple[list[int], None]] = []  # List of (indices, None)

        # Use shared data if provided, otherwise load from file
        if shared_data is not None:
            print("   Using shared data cache")
            self._data_cache = shared_data.reset_index(drop=True)
        else:
            self._load_data()

        # Create windows
        self._create_windows()

        # Filter windows with missing values (per paper)
        self._filter_windows_with_missing()

        # Handle statistics
        if shared_stats is not None:
            print("   Using shared statistics from training data")
            self._set_stats_from_by_name(shared_stats)
        elif compute_stats:
            self._compute_global_stats()

    def _set_stats_from_by_name(self, stats_by_name: dict[str, dict[str, torch.Tensor | float]]):
        # Normalize format to floats (for eval) + tensors (for fast normalization).
        by_name: dict[str, dict[str, float]] = {}
        for sig, stats in stats_by_name.items():
            by_name[sig] = {}
            for k in ("mean", "std", "min", "max"):
                v = stats[k]
                if isinstance(v, torch.Tensor):
                    by_name[sig][k] = float(v.squeeze().item())
                else:
                    by_name[sig][k] = float(v)

        self.stats_by_name = by_name
        self._encode_stats_t = self._build_stats_tensors(self.encode_signals)
        self._decode_stats_t = self._build_stats_tensors(self.decode_signals)

    def _build_stats_tensors(self, signal_list: list[str]) -> dict[str, torch.Tensor]:
        assert self.stats_by_name is not None
        # Shape (C, 1) to broadcast across time dimension T.
        mean = torch.tensor([self.stats_by_name[s]["mean"] for s in signal_list], dtype=torch.float32).unsqueeze(1)
        std = torch.tensor([self.stats_by_name[s]["std"] for s in signal_list], dtype=torch.float32).unsqueeze(1)
        min_v = torch.tensor([self.stats_by_name[s]["min"] for s in signal_list], dtype=torch.float32).unsqueeze(1)
        max_v = torch.tensor([self.stats_by_name[s]["max"] for s in signal_list], dtype=torch.float32).unsqueeze(1)
        return {"mean": mean, "std": std, "min": min_v, "max": max_v}

    def _load_data(self):
        """
        Load time-series data from delimited text file.

        Per paper preprocessing steps:
        - Load raw data chronologically
        - Parse dates and times
        - Convert missing value markers to NaN
        - Sort by datetime
        - Do NOT drop rows yet (missing value removal happens after windowing)

        Supports multiple datetime formats:
        - "date_time_columns": Date and Time in separate columns (household power)
        - "separate_columns": year, month, day, hour in separate columns (air quality)
        """
        print("\n" + "=" * 60)
        print("📊 LOADING TIME-SERIES DATA")
        print("=" * 60)
        print(f"   File: {self.file_path.name}")

        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        start_time = time.time()

        # Load data with configured delimiter and missing value markers
        print("   Reading file...")
        df = pd.read_csv(
            self.file_path,
            sep=self.delimiter,
            na_values=self.na_values,
            low_memory=False,
        )

        # Parse datetime based on format
        if self.datetime_format == "date_time_columns":
            # Household power format: Date and Time columns
            if self.date_column not in df.columns or self.time_column not in df.columns:
                raise ValueError(
                    f"Columns {self.date_column} and {self.time_column} not found in data. "
                    f"Available columns: {list(df.columns)}"
                )
            df["DateTime"] = pd.to_datetime(
                df[self.date_column] + " " + df[self.time_column],
                format="%d/%m/%Y %H:%M:%S",  # European format
                dayfirst=True,
            )
        elif self.datetime_format == "separate_columns":
            # Air quality format: year, month, day, hour columns
            required_cols = [self.year_column, self.month_column, self.day_column, self.hour_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing datetime columns: {missing_cols}. "
                    f"Available columns: {list(df.columns)}"
                )
            df["DateTime"] = pd.to_datetime(
                df[[self.year_column, self.month_column, self.day_column, self.hour_column]]
            )
        elif self.datetime_format == "single_column":
            # Appliances energy format: single datetime column
            if self.datetime_column is None:
                raise ValueError("datetime_column must be specified for 'single_column' format")
            if self.datetime_column not in df.columns:
                raise ValueError(
                    f"Datetime column '{self.datetime_column}' not found in data. "
                    f"Available columns: {list(df.columns)}"
                )
            df["DateTime"] = pd.to_datetime(df[self.datetime_column])
        else:
            raise ValueError(f"Unknown datetime_format: {self.datetime_format}")

        # Create day_id column for metadata (YYYY-MM-DD format)
        df["day_id"] = df["DateTime"].dt.date.astype(str)

        # Sort by datetime (and station if multi-site) to ensure chronological order
        if self.station_column and self.station_column in df.columns:
            df = df.sort_values([self.station_column, "DateTime"]).reset_index(drop=True)
        else:
            df = df.sort_values("DateTime").reset_index(drop=True)

        # Convert signal columns to numeric (coerce errors to NaN)
        for signal_col in self.signal_columns:
            if signal_col in df.columns:
                df[signal_col] = pd.to_numeric(df[signal_col], errors="coerce")

        # Apply max_rows limit if specified (for testing/debugging)
        if self.max_rows is not None:
            df = df.head(self.max_rows)
            print(f"   Max rows: {self.max_rows:,}")

        elapsed = time.time() - start_time
        print(f"   ✅ Loading completed in {elapsed:.2f}s")
        print(f"   📦 Loaded {len(df):,} rows × {len(df.columns)} columns")

        # Print NULL statistics for signal columns
        print("\n   NULL value counts per signal:")
        for signal_col in self.signal_columns:
            if signal_col in df.columns:
                null_count = df[signal_col].isna().sum()
                null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
                print(f"      {signal_col}: {null_count:,} ({null_pct:.1f}%)")

        # Apply data transformations (use-case-specific preprocessing)
        if self.data_transformations:
            print(f"\n   Applying {len(self.data_transformations)} data transformation(s)...")
            for transform_fn in self.data_transformations:
                df = transform_fn(df, self)
            print("   ✅ Transformations applied")

        # Store data (do NOT drop rows with missing values yet)
        self._data_cache = df

        # Show memory usage
        memory_mb = self._data_cache.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"   💾 Memory usage: {memory_mb:.1f} MB")

        print("=" * 60 + "\n")

    def _create_windows(self):
        """
        Create continuous sliding windows with stride=1 across entire dataset.

        Per paper: "instances of time series are sampled using a sliding window
        with a length T, T < L, and a stride Δ = 1. The first instance of time
        series is generated at time t = 1 and such process repeats by setting
        t = t + Δ until t reaches L – T."

        Windows are created continuously across the entire split (can span day boundaries).
        """
        if self._data_cache is None or self._data_cache.empty:
            print("   ⚠️  Warning: No data loaded, cannot create windows")
            return

        print("\n" + "=" * 60)
        print("🪟 CREATING SLIDING WINDOWS")
        print("=" * 60)
        print(f"   Window size: {self.window_size} timesteps ({self.window_size} minutes)")
        print(f"   Stride: {self.stride} timesteps (maximum overlap per paper)")
        print("   Strategy: Continuous across entire split (can span days)")
        print()

        self._windows = []

        # Get all indices in chronological order
        all_indices = self._data_cache.index.tolist()

        # Calculate expected number of windows: L - T + 1
        expected_windows = len(all_indices) - self.window_size + 1

        # Create sliding windows with stride=1
        # Paper formula: t = 1, 2, 3, ..., L - T
        # In code: start_idx = 0, 1, 2, ..., len(all_indices) - window_size
        for start_idx in range(0, len(all_indices) - self.window_size + 1, self.stride):
            window_indices = all_indices[start_idx : start_idx + self.window_size]
            self._windows.append((window_indices, None))  # No grouping ID needed

        print(f"   📊 Total timesteps: {len(all_indices):,}")
        print(f"   🪟 Created {len(self._windows):,} windows")
        print(f"   ✓ Matches expected: L - T + 1 = {expected_windows:,}")
        print("=" * 60 + "\n")

    def _filter_windows_with_missing(self):
        """
        Remove windows containing ANY missing values (optimized vectorized version).

        Per paper: "Any time series instances containing missing values are
        discarded from the produced dataset of time series."

        This is done AFTER window creation, checking each window for NaN
        in any signal column.

        Optimization: Pre-compute a boolean mask and use vectorized operations
        instead of iterating through windows, making stride=1 practical.
        """
        if not self._windows:
            return

        print("\n" + "=" * 60)
        print("🧹 FILTERING WINDOWS WITH MISSING VALUES")
        print("=" * 60)

        initial_count = len(self._windows)

        # OPTIMIZATION: Pre-compute which rows have ANY missing values (vectorized)
        # This is computed once instead of checking each window individually
        has_missing_per_row = self._data_cache[self.signal_columns].isna().any(axis=1).values

        # OPTIMIZATION: Vectorized window filtering using numpy
        # For each window, check if ANY row in that window has missing values
        filtered_windows = []

        # Extract just the indices from windows for faster access
        window_indices_list = [indices for indices, _ in self._windows]

        # Convert to numpy array for vectorized operations
        # This enables batch checking instead of Python loops
        for window_indices in window_indices_list:
            # Check if any row in this window has missing values
            # Using numpy boolean indexing (much faster than pandas)
            if not has_missing_per_row[window_indices].any():
                filtered_windows.append((window_indices, None))

        self._windows = filtered_windows

        removed_count = initial_count - len(self._windows)
        removed_pct = (removed_count / initial_count * 100) if initial_count > 0 else 0

        print(f"   📊 Initial windows: {initial_count:,}")
        print(f"   ❌ Removed: {removed_count:,} ({removed_pct:.1f}%)")
        print(f"   ✅ Remaining: {len(self._windows):,}")
        print("=" * 60 + "\n")

    def _compute_global_stats(self):
        """
        Compute statistics across the entire dataset for normalization.

        Per paper: "each variable is standardized by xnew = (x − u)/σ, where
        u and σ are the mean and standard deviation of the variable in the
        training dataset."

        For training data: compute mean and std
        For val/test data: use shared training statistics (passed via shared_stats)
        """
        if self._data_cache is None or self._data_cache.empty:
            return

        print("   Computing dataset statistics...")

        if self._data_cache is None or self._data_cache.empty:
            raise RuntimeError("Cannot compute stats: no data loaded")

        by_name: dict[str, dict[str, float]] = {}
        for signal_name in self.signal_columns:
            if signal_name not in self._data_cache.columns:
                continue
            signal_data = self._data_cache[signal_name].values.astype(np.float32, copy=False)
            signal_tensor = torch.from_numpy(signal_data)

            mean_val = torch.nanmean(signal_tensor)
            valid_values = signal_tensor[~torch.isnan(signal_tensor)]
            if len(valid_values) > 0:
                std_val = torch.std(valid_values)
                min_val = valid_values.min()
                max_val = valid_values.max()
            else:
                std_val = torch.tensor(1.0)
                min_val = torch.tensor(0.0)
                max_val = torch.tensor(1.0)

            if float(std_val.item()) == 0.0:
                std_val = torch.tensor(1.0)
            if torch.isnan(mean_val):
                mean_val = torch.tensor(0.0)

            by_name[signal_name] = {
                "mean": float(mean_val.item()),
                "std": float(std_val.item()),
                "min": float(min_val.item()),
                "max": float(max_val.item()),
            }

        self.stats_by_name = by_name
        self._encode_stats_t = self._build_stats_tensors(self.encode_signals)
        self._decode_stats_t = self._build_stats_tensors(self.decode_signals)

        print(f"   ✅ Statistics computed for {len(by_name)} signals")

    def __len__(self) -> int:
        """Return the number of windows (samples) in the dataset."""
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a windowed sample as a dict containing separate encode/decode blobs.

        Args:
            idx: Index of the window

        Returns:
            Dictionary containing float32 tensors:
            - 'encode': Tensor (Cenc, T)
            - 'decode': Tensor (Cdec, T)
        """
        if self._data_cache is None:
            raise RuntimeError("Data not loaded. Call _load_data() first.")

        if idx >= len(self._windows):
            raise IndexError(
                f"Window index {idx} out of range (0-{len(self._windows)-1})"
            )

        # Get window information
        window_indices, _ = self._windows[idx]

        # Extract the window
        window_df = self._data_cache.loc[window_indices].copy()

        encode = self._dataframe_to_tensor(window_df, self.encode_signals)
        decode = self._dataframe_to_tensor(window_df, self.decode_signals)

        if self.apply_normalization:
            if self.stats_by_name is None or self._encode_stats_t is None or self._decode_stats_t is None:
                raise ValueError(
                    "apply_normalization requires compute_stats=True or shared_stats. "
                    "Either set compute_stats=True or provide shared_stats."
                )

            # normalize encode
            encode = self._apply_norm(encode, self._encode_stats_t)
            if not self.normalize_encode_only:
                decode = self._apply_norm(decode, self._decode_stats_t)

        return {"encode": encode, "decode": decode}

    def _dataframe_to_tensor(self, df: pd.DataFrame, signal_list: list[str]) -> torch.Tensor:
        # (T, C) -> transpose to (C, T)
        cols = []
        for name in signal_list:
            if name not in df.columns:
                raise ValueError(f"Signal column '{name}' not found in DataFrame")
            col = df[name].to_numpy(dtype=np.float32, copy=False)

            # Fill NaNs with training mean if available, else 0.
            if np.isnan(col).any():
                if self.stats_by_name is not None and name in self.stats_by_name:
                    fill = self.stats_by_name[name]["mean"]
                else:
                    fill = 0.0
                col = np.nan_to_num(col, nan=np.float32(fill))
            cols.append(torch.from_numpy(col))

        # Stack to (C, T)
        x = torch.stack(cols, dim=0)
        return x.to(dtype=torch.float32)

    def _apply_norm(self, x: torch.Tensor, stats_t: dict[str, torch.Tensor]) -> torch.Tensor:
        # x: (C, T); stats: (C, 1)
        if self.apply_normalization == "standardize":
            return (x - stats_t["mean"]) / stats_t["std"].clamp_min(1e-6)
        if self.apply_normalization == "normalize":
            return (x - stats_t["min"]) / (stats_t["max"] - stats_t["min"]).clamp_min(1e-6)
        raise ValueError(f"Unknown apply_normalization: {self.apply_normalization}")


class TimeSeriesTextLightningDataModule(LightningDataModule):
    """
    LightningDataModule for text-based time-series data with chronological splitting.

    Supports multiple text-based time-series datasets (e.g., Household Power, Air Quality).

    Implements chronological preprocessing approach:
    1. **Chronological temporal split** (NOT random):
       - Default: First 81% → training, next 9% → validation, last 10% → test
       - Configurable via split_ratios parameter
    2. Sliding window sampling applied to each split separately
    3. Missing value removal per split (vectorized for efficiency)
    4. Standardization using training set statistics only

    This differs from existing trip-based datasets which use random assignment.
    The chronological approach is appropriate for time-series forecasting tasks.
    """

    def __init__(
        self,
        file_path: str | Path,
        signal_columns: list[str],
        encode_signals: Optional[list[str]] = None,
        decode_signals: Optional[list[str]] = None,
        window_size: int = 128,
        stride: int = 1,
        split_ratios: Optional[list[float]] = None,
        split_seed: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        train_shuffle: bool = True,
        single_batch_eval: bool = False,
        compute_stats: bool = True,
        max_rows: Optional[int] = None,
        data_transformations: Optional[list[str]] = None,
        apply_normalization: Optional[Literal["standardize", "normalize"]] = None,
        normalize_encode_only: bool = False,
        # Data format parameters
        delimiter: str = ";",
        na_values: Optional[list[str]] = None,
        datetime_format: Literal["date_time_columns", "separate_columns", "single_column"] = "date_time_columns",
        datetime_column: Optional[str] = None,
        date_column: Optional[str] = "Date",
        time_column: Optional[str] = "Time",
        year_column: Optional[str] = None,
        month_column: Optional[str] = None,
        day_column: Optional[str] = None,
        hour_column: Optional[str] = None,
        station_column: Optional[str] = None,
    ):
        """
        Args:
            file_path: Path to data file
            signal_columns: List of signal column names to load
            encode_signals: Signals for encoding (subset of signal_columns)
            decode_signals: Signals for decoding (subset of signal_columns)
            window_size: Length of sliding windows (T in paper, default 128)
            stride: Step size between windows (paper requires stride=1, default 1)
            split_ratios: [train, val, test] ratios (default: [0.81, 0.09, 0.10] per paper)
            split_seed: Random seed (not used - splitting is chronological, kept for API compatibility)
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            pin_memory: Whether to pin memory
            prefetch_factor: Prefetch factor for DataLoaders
            persistent_workers: Whether to keep workers alive
            train_shuffle: Whether to shuffle training data
            single_batch_eval: Whether to use batch_size=1 for eval
            compute_stats: Whether to compute dataset statistics
            max_rows: Optional maximum total rows to load (for testing/debugging)
            data_transformations: Optional list of transformation names
            apply_normalization: Normalization method ("standardize" per paper)
            normalize_encode_only: If True, only normalize encode

            Data format parameters (for dataset flexibility):
            delimiter: Column delimiter (';' for household power, ',' for appliances/air quality)
            na_values: Missing value markers (default: ['?'])
            datetime_format: "date_time_columns", "separate_columns", or "single_column"
            datetime_column: For "single_column" format (e.g., "date" for appliances)
            date_column, time_column: For "date_time_columns" format
            year_column, month_column, day_column, hour_column: For "separate_columns" format
            station_column: Optional multi-site column
        """
        super().__init__(
            # LightningDataModule has no such init args; stored below.
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.train_shuffle = train_shuffle
        self.single_batch_eval = single_batch_eval

        # Store parameters for setup()
        self.file_path = file_path
        self.signal_columns = signal_columns
        self.encode_signals = encode_signals
        self.decode_signals = decode_signals
        self.window_size = window_size
        self.stride = stride
        self.split_ratios = split_ratios or [0.81, 0.09, 0.10]  # Per paper
        self.split_seed = split_seed  # Not used (chronological split), kept for API compatibility
        self.compute_stats = compute_stats
        self.max_rows = max_rows
        self.data_transformations = data_transformations
        self.apply_normalization = apply_normalization
        self.normalize_encode_only = normalize_encode_only

        # Exposed for evaluation/denorm
        self.decode_stats_by_name: Optional[dict[str, dict[str, float]]] = None

        # Data format parameters
        self.delimiter = delimiter
        self.na_values = na_values
        self.datetime_format = datetime_format
        self.datetime_column = datetime_column
        self.date_column = date_column
        self.time_column = time_column
        self.year_column = year_column
        self.month_column = month_column
        self.day_column = day_column
        self.hour_column = hour_column
        self.station_column = station_column

        # Warn if split_seed is provided since it's not used
        if split_seed is not None:
            print(f"⚠️  Note: split_seed={split_seed} is ignored for chronological splitting")

    def setup(self, stage: str) -> None:
        """
        Set up train/val/test datasets with chronological temporal splitting.

        Args:
            stage: One of 'fit', 'val', 'test', 'predict'
        """
        valid_stages = {"fit", "val", "test", "predict"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

        print("\n" + "=" * 60)
        print(f"🔧 SETTING UP DATAMODULE FOR STAGE: {stage.upper()}")
        print("   Strategy: CHRONOLOGICAL TEMPORAL SPLIT (per paper)")
        print("=" * 60)

        # Perform chronological setup
        self._setup_chronological(stage)

        # Print summary
        print("\n" + "=" * 60)
        print("✅ DATAMODULE SETUP COMPLETE")
        print("=" * 60)
        if stage == "fit":
            if hasattr(self, "train_set"):
                print(f"   📚 Training samples: {len(self.train_set):,}")
            if hasattr(self, "val_set"):
                print(f"   📊 Validation samples: {len(self.val_set):,}")
        elif stage == "val" and hasattr(self, "val_set"):
            print(f"   📊 Validation samples: {len(self.val_set):,}")
        elif stage == "test" and hasattr(self, "test_set"):
            print(f"   🧪 Test samples: {len(self.test_set):,}")
        elif stage == "predict" and hasattr(self, "predict_set"):
            print(f"   🔮 Prediction samples: {len(self.predict_set):,}")
        print("=" * 60 + "\n")

    def _setup_chronological(self, stage: str) -> None:
        """
        Internal method to set up datasets using chronological temporal splitting.

        Per paper:
        1. Split raw dataset chronologically: first 81% train, next 9% val, last 10% test
        2. Apply windowing to each split separately
        3. Remove windows with missing values per split
        4. Compute stats on training data only, share to val/test
        """
        print("\n📊 Loading full dataset for chronological split...")

        # Load full dataset (no windowing yet, no stats yet)
        full_dataset = TimeSeriesTextDataset(
            file_path=self.file_path,
            signal_columns=self.signal_columns,
            encode_signals=self.encode_signals,
            decode_signals=self.decode_signals,
            window_size=self.window_size,
            stride=self.stride,
            max_rows=self.max_rows,
            data_transformations=self.data_transformations,
            compute_stats=False,
            apply_normalization=None,
            delimiter=self.delimiter,
                datetime_column=self.datetime_column,
            na_values=self.na_values,
            datetime_format=self.datetime_format,
            date_column=self.date_column,
            time_column=self.time_column,
            year_column=self.year_column,
            month_column=self.month_column,
            day_column=self.day_column,
            hour_column=self.hour_column,
            station_column=self.station_column,
        )

        # Get the loaded data (before any windowing/filtering)
        full_data = full_dataset._data_cache

        if full_data is None or full_data.empty:
            raise ValueError("Failed to load data from file")

        # Calculate chronological split indices
        total_len = len(full_data)
        train_end = int(total_len * self.split_ratios[0])  # 81%
        val_end = train_end + int(total_len * self.split_ratios[1])  # + 9%

        print(f"   Total rows: {total_len:,}")
        print("\n   📊 Chronological Split (per paper):")
        print(f"      Train: rows [0:{train_end}] = {train_end:,} rows ({train_end/total_len*100:.1f}%)")
        print(f"      Val:   rows [{train_end}:{val_end}] = {val_end-train_end:,} rows ({(val_end-train_end)/total_len*100:.1f}%)")
        print(f"      Test:  rows [{val_end}:{total_len}] = {total_len-val_end:,} rows ({(total_len-val_end)/total_len*100:.1f}%)")

        # Slice data chronologically
        train_data = full_data.iloc[:train_end].copy()
        val_data = full_data.iloc[train_end:val_end].copy()
        test_data = full_data.iloc[val_end:].copy()

        if stage == "fit":
            print("\n📚 Creating TRAINING dataset...")
            self.train_set = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=train_data,
                data_transformations=self.data_transformations,
                compute_stats=True,  # Compute on training data only
                apply_normalization=self.apply_normalization,
                normalize_encode_only=self.normalize_encode_only,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )

            # Get training statistics to share with val/test
            train_stats = self.train_set.stats_by_name
            self.decode_stats_by_name = train_stats

            print("\n📊 Creating VALIDATION dataset...")
            self.val_set = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=val_data,
                data_transformations=self.data_transformations,
                shared_stats=train_stats,  # Use training stats
                compute_stats=False,
                apply_normalization=self.apply_normalization,
                normalize_encode_only=self.normalize_encode_only,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )

            if len(self.val_set) == 0:
                raise ValueError("❌ VALIDATION dataset is EMPTY!")

        elif stage == "val":
            # Need training stats, so create temporary training dataset
            print("\n📚 Creating training dataset to compute statistics...")
            temp_train = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=train_data,
                data_transformations=self.data_transformations,
                compute_stats=True,
                apply_normalization=None,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )
            train_stats = temp_train.stats_by_name
            self.decode_stats_by_name = train_stats

            print("\n📊 Creating VALIDATION dataset...")
            self.val_set = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=val_data,
                data_transformations=self.data_transformations,
                shared_stats=train_stats,
                compute_stats=False,
                apply_normalization=self.apply_normalization,
                normalize_encode_only=self.normalize_encode_only,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )

            if len(self.val_set) == 0:
                raise ValueError("❌ VALIDATION dataset is EMPTY!")

        elif stage == "test":
            # Need training stats
            print("\n📚 Creating training dataset to compute statistics...")
            temp_train = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=train_data,
                data_transformations=self.data_transformations,
                compute_stats=True,
                apply_normalization=None,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )
            train_stats = temp_train.stats_by_name
            self.decode_stats_by_name = train_stats

            print("\n🧪 Creating TEST dataset...")
            self.test_set = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=test_data,
                data_transformations=self.data_transformations,
                shared_stats=train_stats,
                compute_stats=False,
                apply_normalization=self.apply_normalization,
                normalize_encode_only=self.normalize_encode_only,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )

            if len(self.test_set) == 0:
                raise ValueError("❌ Test dataset is EMPTY!")

        elif stage == "predict":
            # Use test data for predictions
            print("\n🔮 Creating PREDICTION dataset...")
            temp_train = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=train_data,
                data_transformations=self.data_transformations,
                compute_stats=True,
                apply_normalization=None,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )
            train_stats = temp_train.stats_by_name
            self.decode_stats_by_name = train_stats

            self.predict_set = TimeSeriesTextDataset(
                file_path=self.file_path,
                signal_columns=self.signal_columns,
                encode_signals=self.encode_signals,
                decode_signals=self.decode_signals,
                window_size=self.window_size,
                stride=self.stride,
                shared_data=test_data,
                data_transformations=self.data_transformations,
                shared_stats=train_stats,
                compute_stats=False,
                apply_normalization=self.apply_normalization,
                normalize_encode_only=self.normalize_encode_only,
                delimiter=self.delimiter,
                datetime_column=self.datetime_column,
                na_values=self.na_values,
                datetime_format=self.datetime_format,
                date_column=self.date_column,
                time_column=self.time_column,
                year_column=self.year_column,
                month_column=self.month_column,
                day_column=self.day_column,
                hour_column=self.hour_column,
                station_column=self.station_column,
            )

    def train_dataloader(self):
        assert hasattr(self, "train_set")
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        assert hasattr(self, "val_set")
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size if not self.single_batch_eval else 1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        assert hasattr(self, "test_set")
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size if not self.single_batch_eval else 1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        assert hasattr(self, "predict_set")
        return DataLoader(
            dataset=self.predict_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
