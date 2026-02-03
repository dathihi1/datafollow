"""Advanced feature extraction for anomaly detection and autoscaling."""

import numpy as np
import pandas as pd
from scipy.stats import entropy


# Special events dictionary (Domain Knowledge from NASA 1995 logs)
SPECIAL_EVENTS = {
    # US Holidays (Low traffic expected)
    '1995-07-04': {'type': 1, 'name': 'Independence Day', 'impact': 'low_traffic'},
    
    # NASA Space Shuttle STS-70 Mission (High traffic - major event)
    '1995-07-13': {'type': 2, 'name': 'STS-70 Launch', 'impact': 'high_traffic'},
    '1995-07-14': {'type': 2, 'name': 'STS-70 Mission Day 1', 'impact': 'high_traffic'},
    '1995-07-15': {'type': 2, 'name': 'STS-70 Mission Day 2', 'impact': 'high_traffic'},
    '1995-07-16': {'type': 2, 'name': 'STS-70 Mission Day 3', 'impact': 'high_traffic'},
    '1995-07-17': {'type': 2, 'name': 'STS-70 Mission Day 4', 'impact': 'high_traffic'},
    '1995-07-18': {'type': 2, 'name': 'STS-70 Mission Day 5', 'impact': 'high_traffic'},
    '1995-07-19': {'type': 2, 'name': 'STS-70 Mission Day 6', 'impact': 'high_traffic'},
    '1995-07-20': {'type': 2, 'name': 'STS-70 Mission Day 7 + Apollo 11 26th Anniversary', 'impact': 'high_traffic'},
    '1995-07-21': {'type': 2, 'name': 'STS-70 Mission Day 8', 'impact': 'high_traffic'},
    '1995-07-22': {'type': 2, 'name': 'STS-70 Landing', 'impact': 'high_traffic'},
    
    # Hurricane (Missing data period)
    '1995-08-01': {'type': 3, 'name': 'Hurricane Start', 'impact': 'outage'},
    '1995-08-02': {'type': 3, 'name': 'Hurricane Day 2', 'impact': 'outage'},
    '1995-08-03': {'type': 3, 'name': 'Hurricane End', 'impact': 'outage'},
}


class AdvancedFeatureExtractor:
    """Extract advanced features for autoscaling analysis.

    Features:
    - Spike detection (z-score based)
    - Trend indicators
    - Volatility measures
    - Traffic composition metrics
    """

    def __init__(self, spike_threshold: float = 3.0, window: int = 15):
        """Initialize extractor.

        Args:
            spike_threshold: Z-score threshold for spike detection
            window: Window size for calculations
        """
        self.spike_threshold = spike_threshold
        self.window = window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all advanced features.

        Args:
            df: DataFrame with aggregated time series data

        Returns:
            DataFrame with advanced features
        """
        df = df.copy()

        # Spike detection
        df = self._add_spike_features(df)

        # Trend indicators
        df = self._add_trend_features(df)

        # Volatility measures
        df = self._add_volatility_features(df)

        # Rate of change features
        df = self._add_rate_features(df)

        # Special events (NEW - Domain Knowledge)
        df = self._add_event_features(df)

        return df

    def _add_spike_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spike detection features.

        Uses z-score (sigma) method to detect anomalous traffic spikes.

        Args:
            df: DataFrame

        Returns:
            DataFrame with spike features
        """
        if "request_count" not in df.columns:
            return df

        col = "request_count"

        # Rolling z-score - use shift(1) to avoid data leakage
        # At time t, we compare to rolling stats from t-1 and earlier
        rolling_mean = df[col].shift(1).rolling(window=self.window, min_periods=1).mean()
        rolling_std = df[col].shift(1).rolling(window=self.window, min_periods=1).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, float("nan"))

        df["spike_score"] = (df[col] - rolling_mean) / rolling_std
        df["spike_score"] = df["spike_score"].fillna(0)

        # Binary spike indicator
        df["is_spike"] = (df["spike_score"] > self.spike_threshold).astype(int)
        df["is_dip"] = (df["spike_score"] < -self.spike_threshold).astype(int)

        # Spike magnitude (how many sigmas above threshold)
        df["spike_magnitude"] = np.maximum(0, df["spike_score"] - self.spike_threshold)

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features.

        Args:
            df: DataFrame

        Returns:
            DataFrame with trend features
        """
        if "request_count" not in df.columns:
            return df

        col = "request_count"

        # Difference from previous period
        df[f"{col}_diff_1"] = df[col].diff(1)

        # Sign of change (1 = up, 0 = same, -1 = down)
        df[f"{col}_direction"] = np.sign(df[f"{col}_diff_1"])

        # Consecutive increases/decreases
        df[f"{col}_streak"] = self._calculate_streak(df[f"{col}_direction"])

        # Trend strength (slope of linear regression over window)
        df[f"{col}_trend"] = self._calculate_trend(df[col], self.window)

        # Compare to same time yesterday (if enough data)
        periods_per_day = 24 * 60 // self._get_minutes_per_period(df)
        if len(df) > periods_per_day:
            df[f"{col}_vs_yesterday"] = df[col] - df[col].shift(periods_per_day)
            df[f"{col}_vs_yesterday_pct"] = df[col].pct_change(periods=periods_per_day)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures.

        Args:
            df: DataFrame

        Returns:
            DataFrame with volatility features
        """
        if "request_count" not in df.columns:
            return df

        col = "request_count"

        # Rolling coefficient of variation - use shift(1) to avoid data leakage
        # At time t, volatility measures use only data from t-1 and earlier
        rolling_mean = df[col].shift(1).rolling(window=self.window, min_periods=1).mean()
        rolling_std = df[col].shift(1).rolling(window=self.window, min_periods=1).std()

        df[f"{col}_cv"] = rolling_std / rolling_mean.replace(0, float("nan"))
        df[f"{col}_cv"] = df[f"{col}_cv"].fillna(0)

        # Bollinger Bands width (volatility indicator)
        df[f"{col}_bb_upper"] = rolling_mean + 2 * rolling_std
        df[f"{col}_bb_lower"] = rolling_mean - 2 * rolling_std
        df[f"{col}_bb_width"] = df[f"{col}_bb_upper"] - df[f"{col}_bb_lower"]

        # Percentage of historical max - uses shift(1) to avoid data leakage
        # At time t, we compare to max seen BEFORE t (not including t)
        expanding_max = df[col].shift(1).expanding().max()
        df[f"{col}_pct_of_max"] = df[col] / expanding_max.replace(0, float("nan"))
        df[f"{col}_pct_of_max"] = df[f"{col}_pct_of_max"].fillna(0)

        return df

    def _add_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change features.

        Args:
            df: DataFrame

        Returns:
            DataFrame with rate features
        """
        if "request_count" not in df.columns:
            return df

        col = "request_count"

        # Acceleration (second derivative)
        df[f"{col}_velocity"] = df[col].diff(1)
        df[f"{col}_acceleration"] = df[f"{col}_velocity"].diff(1)

        # Momentum (sum of recent changes)
        df[f"{col}_momentum"] = df[f"{col}_velocity"].rolling(window=5, min_periods=1).sum()

        return df

    def _calculate_streak(self, direction: pd.Series) -> pd.Series:
        """Calculate consecutive streak of increases/decreases.

        Args:
            direction: Series of -1, 0, 1 values

        Returns:
            Series of streak lengths (positive for up, negative for down)
        """
        # Group consecutive same directions
        change = direction != direction.shift(1)
        groups = change.cumsum()

        # Count within each group
        streak = direction.groupby(groups).cumsum()

        return streak

    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear trend.

        Args:
            series: Time series
            window: Window size

        Returns:
            Series of trend slopes
        """
        def linear_slope(x):
            if len(x) < 2:
                return 0
            x_vals = np.arange(len(x))
            coeffs = np.polyfit(x_vals, x, 1)
            return coeffs[0]

        return series.rolling(window=window, min_periods=2).apply(linear_slope, raw=True)

    def _get_minutes_per_period(self, df: pd.DataFrame) -> int:
        """Estimate minutes per period from data.

        Args:
            df: DataFrame with timestamp

        Returns:
            Minutes per period
        """
        if "timestamp" not in df.columns:
            return 5  # Default

        if len(df) < 2:
            return 5

        timestamps = pd.to_datetime(df["timestamp"])
        diff = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 60
        return max(1, int(diff))

    def _add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add special event features based on domain knowledge.
        
        Uses SPECIAL_EVENTS dictionary to identify known events
        (holidays, NASA missions, outages) and their expected impact.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with event features
        """
        if "timestamp" not in df.columns:
            return df
            
        # Initialize event columns
        df['event_type'] = 0  # 0: normal, 1: holiday, 2: space event, 3: outage
        df['is_special_event'] = 0
        df['event_impact'] = 'normal'
        df['event_name'] = ''
        
        # Map events
        for date_str, event_info in SPECIAL_EVENTS.items():
            try:
                event_date = pd.to_datetime(date_str).date()
                mask = df['timestamp'].dt.date == event_date
                
                if mask.any():
                    df.loc[mask, 'event_type'] = event_info['type']
                    df.loc[mask, 'is_special_event'] = 1
                    df.loc[mask, 'event_impact'] = event_info['impact']
                    df.loc[mask, 'event_name'] = event_info['name']
            except Exception:
                continue  # Skip invalid dates
        
        return df

    def calculate_host_entropy(
        self, hosts: pd.Series, normalize: bool = True
    ) -> float:
        """Calculate entropy of host distribution.

        High entropy = many different hosts (normal traffic)
        Low entropy = few hosts dominating (possible DDoS)

        Args:
            hosts: Series of host identifiers
            normalize: Whether to normalize by max entropy

        Returns:
            Entropy value
        """
        value_counts = hosts.value_counts()
        probabilities = value_counts / value_counts.sum()

        ent = entropy(probabilities)

        if normalize:
            max_entropy = np.log(len(value_counts))
            ent = ent / max_entropy if max_entropy > 0 else 0

        return ent

    def get_feature_names(self) -> list[str]:
        """Get list of feature names produced by this extractor.

        Returns:
            List of feature names
        """
        return [
            "spike_score", "is_spike", "is_dip", "spike_magnitude",
            "request_count_diff_1", "request_count_direction", "request_count_streak",
            "request_count_trend", "request_count_vs_yesterday", "request_count_vs_yesterday_pct",
            "request_count_cv", "request_count_bb_upper", "request_count_bb_lower",
            "request_count_bb_width", "request_count_pct_of_max",
            "request_count_velocity", "request_count_acceleration", "request_count_momentum",
        ]
