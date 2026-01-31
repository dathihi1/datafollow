"""Time aggregation module for converting raw logs to time series."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class AggregationConfig:
    """Configuration for time aggregation."""

    window: str  # pandas offset alias: '1min', '5min', '15min'
    fill_method: Literal["zero", "ffill", "interpolate"] = "zero"
    include_host_stats: bool = True
    include_error_stats: bool = True
    include_bytes_stats: bool = True


class TimeAggregator:
    """Aggregate raw log entries into time series data.

    Converts individual log entries into fixed time intervals with
    computed statistics for each window.
    """

    # Default aggregation windows
    WINDOW_1M = "1min"
    WINDOW_5M = "5min"
    WINDOW_15M = "15min"

    def __init__(self, config: AggregationConfig | None = None):
        """Initialize aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig(window=self.WINDOW_5M)

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate DataFrame to time series.

        Args:
            df: Cleaned DataFrame with timestamp column

        Returns:
            Aggregated DataFrame with one row per time window
        """
        df = df.copy()

        # Ensure timestamp is the index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Basic aggregations
        agg_dict = {
            "host": "count",  # request_count
        }

        # Host statistics
        if self.config.include_host_stats:
            agg_dict["host"] = ["count", "nunique"]

        # Error statistics
        if self.config.include_error_stats and "is_error" in df.columns:
            agg_dict["is_error"] = ["sum", "mean"]

        if self.config.include_error_stats and "is_success" in df.columns:
            agg_dict["is_success"] = "mean"

        # Bytes statistics
        if self.config.include_bytes_stats and "bytes" in df.columns:
            agg_dict["bytes"] = ["sum", "mean", "max"]

        # Perform aggregation
        result = df.resample(self.config.window).agg(agg_dict)

        # Flatten column names
        result.columns = self._flatten_columns(result.columns)

        # Rename columns for clarity
        rename_map = {
            "host_count": "request_count",
            "host_nunique": "unique_hosts",
            "is_error_sum": "error_count",
            "is_error_mean": "error_rate",
            "is_success_mean": "success_rate",
            "bytes_sum": "bytes_total",
            "bytes_mean": "bytes_avg",
            "bytes_max": "bytes_max",
        }
        result = result.rename(columns=rename_map)

        # Fill missing periods
        result = self._fill_missing(result)

        # Add derived metrics
        result = self._add_derived_metrics(result)

        result = result.reset_index()
        if "index" in result.columns:
            result = result.rename(columns={"index": "timestamp"})
        return result

    def _flatten_columns(self, columns: pd.MultiIndex) -> list[str]:
        """Flatten MultiIndex columns to single level.

        Args:
            columns: MultiIndex columns

        Returns:
            List of flattened column names
        """
        if isinstance(columns, pd.MultiIndex):
            return ["_".join(col).strip("_") for col in columns.values]
        return list(columns)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing time periods.

        Args:
            df: Aggregated DataFrame

        Returns:
            DataFrame with filled missing periods
        """
        # Create complete time index
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=self.config.window,
        )

        # Reindex to include all periods
        df = df.reindex(full_index)

        # Fill missing values
        if self.config.fill_method == "zero":
            df = df.fillna(0)
        elif self.config.fill_method == "ffill":
            df = df.ffill()
        elif self.config.fill_method == "interpolate":
            df = df.interpolate(method="linear")

        # Ensure counts are integers
        count_cols = [col for col in df.columns if "count" in col.lower()]
        for col in count_cols:
            df[col] = df[col].astype(int)

        return df

    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics to aggregated data.

        Args:
            df: Aggregated DataFrame

        Returns:
            DataFrame with additional metrics
        """
        # Requests per unique host (engagement metric)
        if "request_count" in df.columns and "unique_hosts" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["requests_per_host"] = df["request_count"] / df["unique_hosts"].replace(0, np.nan)
            df["requests_per_host"] = df["requests_per_host"].fillna(0)

        # Bytes per request
        if "bytes_total" in df.columns and "request_count" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["bytes_per_request"] = df["bytes_total"] / df["request_count"].replace(0, np.nan)
            df["bytes_per_request"] = df["bytes_per_request"].fillna(0)

        return df

    def aggregate_multiple_windows(
        self, df: pd.DataFrame, windows: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """Aggregate to multiple time windows.

        Args:
            df: Cleaned DataFrame
            windows: List of window sizes (default: 1min, 5min, 15min)

        Returns:
            Dictionary mapping window name to aggregated DataFrame
        """
        windows = windows or [self.WINDOW_1M, self.WINDOW_5M, self.WINDOW_15M]
        results = {}

        for window in windows:
            config = AggregationConfig(
                window=window,
                fill_method=self.config.fill_method,
                include_host_stats=self.config.include_host_stats,
                include_error_stats=self.config.include_error_stats,
                include_bytes_stats=self.config.include_bytes_stats,
            )
            aggregator = TimeAggregator(config)
            results[window] = aggregator.aggregate(df)
            print(f"Aggregated to {window}: {len(results[window]):,} rows")

        return results

    def save(
        self, df: pd.DataFrame, filepath: str | Path, format: str = "parquet"
    ) -> None:
        """Save aggregated DataFrame.

        Args:
            df: Aggregated DataFrame
            filepath: Output path
            format: Output format
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            df.to_parquet(filepath, index=False, engine="pyarrow")
        elif format == "csv":
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved {len(df):,} rows to {filepath}")

    @staticmethod
    def load(filepath: str | Path) -> pd.DataFrame:
        """Load aggregated DataFrame.

        Args:
            filepath: Path to saved file

        Returns:
            DataFrame
        """
        filepath = Path(filepath)

        if filepath.suffix == ".parquet":
            return pd.read_parquet(filepath)
        elif filepath.suffix == ".csv":
            return pd.read_csv(filepath, parse_dates=["timestamp"])
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

    def get_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics for aggregated data.

        Args:
            df: Aggregated DataFrame

        Returns:
            Statistics dictionary
        """
        timestamp_col = "timestamp" if "timestamp" in df.columns else df.index

        return {
            "rows": len(df),
            "window": self.config.window,
            "date_range": {
                "start": pd.Series(timestamp_col).min(),
                "end": pd.Series(timestamp_col).max(),
            },
            "request_stats": {
                "total": df["request_count"].sum() if "request_count" in df.columns else 0,
                "mean": df["request_count"].mean() if "request_count" in df.columns else 0,
                "max": df["request_count"].max() if "request_count" in df.columns else 0,
                "std": df["request_count"].std() if "request_count" in df.columns else 0,
            },
            "missing_periods": (df["request_count"] == 0).sum() if "request_count" in df.columns else 0,
        }
