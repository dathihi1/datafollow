"""Data cleaning module for NASA web server logs."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""

    original_rows: int
    final_rows: int
    duplicates_removed: int
    invalid_status_removed: int
    negative_bytes_fixed: int
    missing_periods: list[tuple[datetime, datetime]]
    outliers_detected: int

    def __str__(self) -> str:
        return (
            f"Cleaning Report:\n"
            f"  Original rows: {self.original_rows:,}\n"
            f"  Final rows: {self.final_rows:,}\n"
            f"  Duplicates removed: {self.duplicates_removed:,}\n"
            f"  Invalid status removed: {self.invalid_status_removed:,}\n"
            f"  Negative bytes fixed: {self.negative_bytes_fixed:,}\n"
            f"  Missing periods: {len(self.missing_periods)}\n"
            f"  Outliers detected: {self.outliers_detected:,}"
        )


class DataCleaner:
    """Cleaner for parsed log data.

    Handles:
    - Duplicate removal
    - Invalid status code handling
    - Missing data period detection (Aug 1-3 hurricane gap)
    - Bytes validation
    - Outlier detection
    """

    # Known missing data period (hurricane)
    MISSING_START = datetime(1995, 8, 1, 14, 52, tzinfo=None)
    MISSING_END = datetime(1995, 8, 3, 4, 36, tzinfo=None)

    # Valid HTTP status codes
    VALID_STATUS_CODES = {100, 101, 200, 201, 202, 204, 206, 301, 302, 303, 304, 307, 308,
                          400, 401, 403, 404, 405, 408, 410, 413, 414, 500, 501, 502, 503, 504}

    def __init__(self, remove_duplicates: bool = True, validate_status: bool = True):
        """Initialize cleaner.

        Args:
            remove_duplicates: Whether to remove duplicate entries
            validate_status: Whether to validate HTTP status codes
        """
        self.remove_duplicates = remove_duplicates
        self.validate_status = validate_status

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """Clean the DataFrame.

        Args:
            df: Raw parsed DataFrame

        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        original_rows = len(df)
        df = df.copy()

        # 1. Remove duplicates (exact same request at same time)
        duplicates_before = len(df)
        if self.remove_duplicates:
            df = df.drop_duplicates(subset=["host", "timestamp", "url", "method"])
        duplicates_removed = duplicates_before - len(df)

        # 2. Validate and fix status codes
        invalid_status_removed = 0
        if self.validate_status:
            invalid_mask = ~df["status"].isin(self.VALID_STATUS_CODES)
            invalid_status_removed = invalid_mask.sum()
            # Keep invalid status but flag them (don't remove, might be important)
            df["status_valid"] = ~invalid_mask
        else:
            df["status_valid"] = True

        # 3. Fix negative bytes (set to 0)
        negative_mask = df["bytes"] < 0
        negative_bytes_fixed = negative_mask.sum()
        df.loc[negative_mask, "bytes"] = 0

        # 4. Ensure timestamp is timezone-naive for easier handling
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # 5. Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 6. Detect missing periods (gaps > 1 hour)
        missing_periods = self._detect_missing_periods(df)

        # 7. Detect outliers in bytes (IQR method)
        outliers_detected = self._detect_outliers(df)

        # 8. Add derived columns
        df = self._add_derived_columns(df)

        report = CleaningReport(
            original_rows=original_rows,
            final_rows=len(df),
            duplicates_removed=duplicates_removed,
            invalid_status_removed=invalid_status_removed,
            negative_bytes_fixed=negative_bytes_fixed,
            missing_periods=missing_periods,
            outliers_detected=outliers_detected,
        )

        return df, report

    def _detect_missing_periods(
        self, df: pd.DataFrame, gap_threshold_hours: float = 1.0
    ) -> list[tuple[datetime, datetime]]:
        """Detect gaps in data longer than threshold.

        Args:
            df: DataFrame sorted by timestamp
            gap_threshold_hours: Minimum gap duration to report

        Returns:
            List of (start, end) tuples for missing periods
        """
        if len(df) < 2:
            return []

        timestamps = df["timestamp"].values
        gaps = []

        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, "h")
            if diff > gap_threshold_hours:
                gaps.append((
                    pd.Timestamp(timestamps[i - 1]).to_pydatetime(),
                    pd.Timestamp(timestamps[i]).to_pydatetime(),
                ))

        return gaps

    def _detect_outliers(self, df: pd.DataFrame, column: str = "bytes") -> int:
        """Detect outliers using IQR method.

        Args:
            df: DataFrame
            column: Column to check for outliers

        Returns:
            Number of outliers detected
        """
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        df["bytes_outlier"] = outlier_mask

        return outlier_mask.sum()

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns.

        Args:
            df: DataFrame

        Returns:
            DataFrame with additional columns
        """
        # Status categories
        df["status_category"] = pd.cut(
            df["status"],
            bins=[0, 199, 299, 399, 499, 599],
            labels=["1xx", "2xx", "3xx", "4xx", "5xx"],
        )

        # Is error
        df["is_error"] = df["status"] >= 400

        # Is success
        df["is_success"] = df["status"] == 200

        # File extension from URL
        df["extension"] = df["url"].str.extract(r"\.([a-zA-Z0-9]+)(?:\?.*)?$")[0].fillna("none")
        df["extension"] = df["extension"].str.lower().astype("category")

        # URL category (based on common NASA paths)
        df["url_category"] = self._categorize_url(df["url"])

        return df

    def _categorize_url(self, urls: pd.Series) -> pd.Series:
        """Categorize URLs into content types.

        Args:
            urls: Series of URL strings

        Returns:
            Series of category labels
        """
        categories = pd.Series("other", index=urls.index)

        # Define patterns and their categories
        patterns = [
            (r"^/images/", "images"),
            (r"^/shuttle/", "shuttle"),
            (r"^/history/", "history"),
            (r"^/software/", "software"),
            (r"^/cgi-bin/", "cgi"),
            (r"^/facilities/", "facilities"),
            (r"^/ksc\.html", "homepage"),
            (r"^/$", "homepage"),
        ]

        for pattern, category in patterns:
            mask = urls.str.contains(pattern, regex=True, na=False)
            categories.loc[mask] = category

        return categories.astype("category")

    def save(self, df: pd.DataFrame, filepath: str | Path, format: str = "parquet") -> None:
        """Save cleaned DataFrame.

        Args:
            df: Cleaned DataFrame
            filepath: Output path
            format: Output format ('parquet' or 'csv')
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
        """Load cleaned DataFrame.

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
