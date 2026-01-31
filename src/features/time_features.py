"""Time-based feature extraction for time series analysis."""

import numpy as np
import pandas as pd


class TimeFeatureExtractor:
    """Extract time-based features from timestamp column.

    Features extracted:
    - hour, minute, day, month, year
    - day_of_week, is_weekend
    - is_business_hour
    - time_of_day (morning/afternoon/evening/night)
    - cyclical encodings (sin/cos)
    """

    def __init__(self, cyclical: bool = True):
        """Initialize extractor.

        Args:
            cyclical: Whether to include cyclical encodings (sin/cos)
        """
        self.cyclical = cyclical

    def transform(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """Extract time features from DataFrame.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with added time features
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        ts = df[timestamp_col]

        # Basic time components
        df["year"] = ts.dt.year
        df["month"] = ts.dt.month
        df["day"] = ts.dt.day
        df["hour"] = ts.dt.hour
        df["minute"] = ts.dt.minute
        df["day_of_week"] = ts.dt.dayofweek  # Monday=0, Sunday=6
        df["day_of_year"] = ts.dt.dayofyear
        df["week_of_year"] = ts.dt.isocalendar().week.astype(int)

        # Weekend flag
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Business hours (8 AM - 6 PM EST)
        df["is_business_hour"] = ((df["hour"] >= 8) & (df["hour"] < 18)).astype(int)

        # Peak hours (11 AM - 5 PM based on EDA)
        df["is_peak_hour"] = ((df["hour"] >= 11) & (df["hour"] < 17)).astype(int)

        # Time of day categories
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 5, 11, 17, 21, 24],
            labels=["night", "morning", "afternoon", "evening", "night_late"],
        )

        # Part of day (simpler version)
        df["part_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
        )

        # Cyclical encodings (for models that need continuous features)
        if self.cyclical:
            df = self._add_cyclical_features(df)

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical sin/cos encodings for time features.

        This helps models understand that hour 23 is close to hour 0.

        Args:
            df: DataFrame with time features

        Returns:
            DataFrame with cyclical features
        """
        # Hour cyclical (24-hour cycle)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week cyclical (7-day cycle)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Day of month cyclical (31-day cycle)
        df["dom_sin"] = np.sin(2 * np.pi * df["day"] / 31)
        df["dom_cos"] = np.cos(2 * np.pi * df["day"] / 31)

        # Month cyclical (12-month cycle)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Minute of day (0-1439)
        minute_of_day = df["hour"] * 60 + df["minute"]
        df["mod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
        df["mod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of feature names produced by this extractor.

        Returns:
            List of feature names
        """
        features = [
            "year", "month", "day", "hour", "minute",
            "day_of_week", "day_of_year", "week_of_year",
            "is_weekend", "is_business_hour", "is_peak_hour",
            "time_of_day", "part_of_day",
        ]

        if self.cyclical:
            features.extend([
                "hour_sin", "hour_cos",
                "dow_sin", "dow_cos",
                "dom_sin", "dom_cos",
                "month_sin", "month_cos",
                "mod_sin", "mod_cos",
            ])

        return features
