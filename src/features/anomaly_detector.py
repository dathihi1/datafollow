"""Anomaly detection for traffic analysis using Isolation Forest."""

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    IsolationForest = None
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed. Install with: pip install scikit-learn")


class TrafficAnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest.
    
    Use cases:
    - DDoS attack detection
    - Unusual traffic spikes
    - System failures
    - Bot/crawler detection
    
    Features compared to Z-score method:
    - Multi-dimensional: Uses multiple features simultaneously
    - Non-parametric: No assumption of normal distribution
    - Unsupervised: No labeled data needed
    - Robust: Less sensitive to outliers in training
    """

    def __init__(
        self,
        contamination: float = 0.01,
        random_state: int = 42,
        n_estimators: int = 100,
        max_samples: str | int = 'auto',
    ):
        """Initialize anomaly detector.

        Args:
            contamination: Expected proportion of outliers (default 1%)
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None):
        """Train IsolationForest on features.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features (for logging)

        Returns:
            Self for method chaining
        """
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            n_jobs=-1,
            verbose=0,
        )

        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            self.feature_names = feature_names
            X_array = X

        self.model.fit(X_array)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Detect anomalies.

        Args:
            X: Feature matrix

        Returns:
            Binary predictions: -1 for anomalies, 1 for normal points
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def decision_function(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Get anomaly scores.

        Lower scores = more anomalous

        Args:
            X: Feature matrix

        Returns:
            Anomaly scores (lower is more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.decision_function(X)

    def fit_predict(self, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        """Fit and predict in one step.

        Args:
            X: Feature matrix
            feature_names: Names of features

        Returns:
            Binary predictions: -1 for anomalies, 1 for normal
        """
        return self.fit(X, feature_names).predict(X)

    def get_anomaly_info(self, df: pd.DataFrame, predictions=None, anomaly_col: str = 'is_anomaly_ml') -> dict:
        """Get timestamps and values of detected anomalies.

        Args:
            df: DataFrame with timestamp and request_count columns
            predictions: Optional numpy array of predictions (-1 = anomaly, 1 = normal)
                        If None, uses anomaly_col from df
            anomaly_col: Name of anomaly indicator column (used if predictions is None)

        Returns:
            Dict mapping timestamp -> request_count for anomalies
        """
        # Handle both prediction array and column name inputs
        if predictions is not None:
            # Convert predictions to boolean mask
            anomaly_mask = (predictions == -1)
        elif anomaly_col in df.columns:
            anomaly_mask = (df[anomaly_col] == 1)
        else:
            raise ValueError(f"Either provide predictions array or ensure '{anomaly_col}' column exists")

        # Get anomaly timestamps and values
        anomalies = df[anomaly_mask]
        
        if 'timestamp' not in df.columns or 'request_count' not in df.columns:
            # Return basic count if columns missing
            return {'anomaly_count': int(anomaly_mask.sum())}
        
        result = {}
        for _, row in anomalies.iterrows():
            result[row['timestamp']] = row['request_count']
        
        return result


def detect_anomalies_multimethod(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.01,
) -> pd.DataFrame:
    """Detect anomalies using both Z-score and Isolation Forest.

    Args:
        df: DataFrame with features
        feature_cols: Columns to use for detection
        contamination: Expected anomaly proportion

    Returns:
        DataFrame with anomaly indicators and scores
    """
    df = df.copy()

    # Initialize detector
    detector = TrafficAnomalyDetector(contamination=contamination)

    # Fit and predict
    X = df[feature_cols].fillna(0)  # Handle any NaN
    predictions = detector.fit_predict(X)
    scores = detector.decision_function(X)

    # Add to DataFrame
    df['is_anomaly_ml'] = (predictions == -1).astype(int)
    df['anomaly_score_ml'] = scores

    # Compare with Z-score if available
    if 'is_spike' in df.columns:
        df['anomaly_agreement'] = (
            (df['is_spike'] == 1) & (df['is_anomaly_ml'] == 1)
        ).astype(int)

    return df
