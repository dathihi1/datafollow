"""Lag feature extraction for time series forecasting."""

import pandas as pd


class LagFeatureExtractor:
    """Extract lag features for time series analysis.

    Creates lagged versions of target and other columns to capture
    temporal dependencies.
    """

    # Default lag periods for different time windows
    DEFAULT_LAGS = {
        "1min": [1, 5, 10, 30, 60, 1440],  # 1m, 5m, 10m, 30m, 1h, 1d
        "5min": [1, 3, 6, 12, 60, 288],    # 5m, 15m, 30m, 1h, 5h, 1d
        "15min": [1, 2, 4, 8, 24, 96],     # 15m, 30m, 1h, 2h, 6h, 1d
    }

    def __init__(self, lags: list[int] | None = None, window: str = "5min"):
        """Initialize extractor.

        Args:
            lags: List of lag periods (if None, uses defaults for window)
            window: Time window for default lag selection
        """
        self.lags = lags or self.DEFAULT_LAGS.get(window, [1, 5, 12, 60, 288])
        self.window = window

    def transform(
        self,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract lag features.

        Args:
            df: DataFrame with time series data
            target_cols: Columns to create lags for (default: request_count, bytes_total)

        Returns:
            DataFrame with lag features
        """
        df = df.copy()

        target_cols = target_cols or ["request_count", "bytes_total"]
        target_cols = [col for col in target_cols if col in df.columns]

        for col in target_cols:
            for lag in self.lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    def transform_with_diff(
        self,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Extract lag features with differences.

        Creates both lag and difference (change from lag) features.

        Args:
            df: DataFrame with time series data
            target_cols: Columns to create features for

        Returns:
            DataFrame with lag and diff features
        """
        df = df.copy()

        target_cols = target_cols or ["request_count"]
        target_cols = [col for col in target_cols if col in df.columns]

        for col in target_cols:
            for lag in self.lags:
                # Lag value
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df[col].shift(lag)

                # Difference from lag
                df[f"{col}_diff_{lag}"] = df[col] - df[lag_col]

                # Percentage change from lag
                df[f"{col}_pct_change_{lag}"] = df[col].pct_change(periods=lag)

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
            for lag in self.lags:
                features.append(f"{col}_lag_{lag}")

        return features
