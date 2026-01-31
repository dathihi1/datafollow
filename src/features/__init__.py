"""Feature engineering modules for time series analysis."""

from src.features.time_features import TimeFeatureExtractor
from src.features.lag_features import LagFeatureExtractor
from src.features.rolling_features import RollingFeatureExtractor
from src.features.advanced_features import AdvancedFeatureExtractor, SPECIAL_EVENTS

# Anomaly detection (optional import)
try:
    from src.features.anomaly_detector import (
        TrafficAnomalyDetector,
        detect_anomalies_multimethod,
    )
    ANOMALY_AVAILABLE = True
except ImportError:
    TrafficAnomalyDetector = None
    detect_anomalies_multimethod = None
    ANOMALY_AVAILABLE = False

__all__ = [
    "TimeFeatureExtractor",
    "LagFeatureExtractor",
    "RollingFeatureExtractor",
    "AdvancedFeatureExtractor",
    "SPECIAL_EVENTS",
    "TrafficAnomalyDetector",
    "detect_anomalies_multimethod",
    "ANOMALY_AVAILABLE",
]
