"""Rolling window statistics for time series analysis."""

import pandas as pd


class RollingFeatureExtractor:
    """Extract rolling window statistics.

    Computes moving averages, standard deviations, min/max, and other
    statistics over rolling windows.
    """

    # Default rolling windows for different time aggregations (optimized)
    DEFAULT_WINDOWS = {
        "1min": [5, 15, 60],       # 5m, 15m, 1h
        "5min": [3, 12, 60],       # 15m, 1h, 5h
        "15min": [4, 16, 96],      # 1h, 4h, 24h
    }

    def __init__(
        self,
        windows: list[int] | None = None,
        time_window: str = "5min",
        min_periods: int = 1,
    ):
        """Initialize extractor.

        Args:
            windows: List of rolling window sizes
            time_window: Time aggregation window for default selection
            min_periods: Minimum periods for rolling calculation
        """
        self.windows = windows or self.DEFAULT_WINDOWS.get(time_window, [5, 15, 30, 60])
        self.time_window = time_window
        self.min_periods = min_periods

    def transform(
        self,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract rolling statistics.

        Args:
            df: DataFrame with time series data
            target_cols: Columns to compute rolling stats for

        Returns:
            DataFrame with rolling features

        Note:
            All rolling features use shift(1) to avoid data leakage.
            This ensures features at time t only use data from t-1 and earlier.
        """
        df = df.copy()

        target_cols = target_cols or ["request_count", "bytes_total"]
        target_cols = [col for col in target_cols if col in df.columns]

        for col in target_cols:
            for window in self.windows:
                # Use shift(1) to avoid data leakage - only use past observations
                rolling = df[col].shift(1).rolling(window=window, min_periods=self.min_periods)

                # Mean (most important)
                df[f"{col}_rolling_mean_{window}"] = rolling.mean()

                # Standard deviation (volatility)
                df[f"{col}_rolling_std_{window}"] = rolling.std()

                # Only compute min/max/range for smaller windows to save time
                if window <= 12:
                    df[f"{col}_rolling_min_{window}"] = rolling.min()
                    df[f"{col}_rolling_max_{window}"] = rolling.max()
                    df[f"{col}_rolling_range_{window}"] = (
                        df[f"{col}_rolling_max_{window}"] - df[f"{col}_rolling_min_{window}"]
                    )

        return df

    def transform_extended(
        self,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract extended rolling statistics.

        Includes quantiles, skewness, and kurtosis.

        Args:
            df: DataFrame with time series data
            target_cols: Columns to compute rolling stats for

        Returns:
            DataFrame with extended rolling features
        """
        df = self.transform(df, target_cols)

        target_cols = target_cols or ["request_count"]
        target_cols = [col for col in target_cols if col in df.columns]

        for col in target_cols:
            for window in self.windows:
                # Use shift(1) to avoid data leakage - only use past observations
                rolling = df[col].shift(1).rolling(window=window, min_periods=self.min_periods)

                # Median (50th percentile)
                df[f"{col}_rolling_median_{window}"] = rolling.median()

                # 25th and 75th percentiles
                df[f"{col}_rolling_q25_{window}"] = rolling.quantile(0.25)
                df[f"{col}_rolling_q75_{window}"] = rolling.quantile(0.75)

                # Coefficient of variation (std / mean)
                mean = df[f"{col}_rolling_mean_{window}"]
                std = df[f"{col}_rolling_std_{window}"]
                df[f"{col}_rolling_cv_{window}"] = std / mean.replace(0, float("nan"))

        return df

    def transform_ewm(
        self,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
        spans: list[int] | None = None,
    ) -> pd.DataFrame:
        """Extract exponentially weighted moving statistics.

        EWM gives more weight to recent observations.

        Args:
            df: DataFrame with time series data
            target_cols: Columns to compute EWM for
            spans: EWM spans (default: [5, 15, 30])

        Returns:
            DataFrame with EWM features
        """
        df = df.copy()

        target_cols = target_cols or ["request_count"]
        target_cols = [col for col in target_cols if col in df.columns]
        spans = spans or [5, 15, 30]

        for col in target_cols:
            for span in spans:
                # Use shift(1) to avoid data leakage - EWM must not include current observation
                ewm = df[col].shift(1).ewm(span=span, min_periods=self.min_periods)

                df[f"{col}_ewm_mean_{span}"] = ewm.mean()
                df[f"{col}_ewm_std_{span}"] = ewm.std()

        return df

    def get_feature_names(self, target_cols: list[str] | None = None) -> list[str]:
        """Get list of feature names produced by this extractor.

        Args:
            target_cols: Target columns

        Returns:
            List of feature names
        """
        target_cols = target_cols or ["request_count", "bytes_total"]
        features = []

        for col in target_cols:
            for window in self.windows:
                features.extend([
                    f"{col}_rolling_mean_{window}",
                    f"{col}_rolling_std_{window}",
                ])
                # min/max/range only computed for smaller windows (see transform())
                if window <= 12:
                    features.extend([
                        f"{col}_rolling_min_{window}",
                        f"{col}_rolling_max_{window}",
                        f"{col}_rolling_range_{window}",
                    ])

        return features
