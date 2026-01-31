"""Tests for anomaly detection module."""

import numpy as np
import pandas as pd
import pytest

from src.features.anomaly_detector import (
    TrafficAnomalyDetector,
    detect_anomalies_multimethod,
    SKLEARN_AVAILABLE,
)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
class TestTrafficAnomalyDetector:
    """Tests for TrafficAnomalyDetector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample traffic data with known anomalies."""
        np.random.seed(42)
        n_samples = 1000
        
        # Normal traffic
        normal = np.random.normal(100, 20, n_samples - 10)
        
        # Add 10 clear anomalies
        anomalies = np.array([500, 550, 480, 520, 510, 490, 530, 505, 515, 525])
        
        data = np.concatenate([normal, anomalies])
        np.random.shuffle(data)
        
        df = pd.DataFrame({
            'request_count': data,
            'unique_hosts': data * 0.5 + np.random.normal(0, 5, n_samples),
            'error_rate': np.random.uniform(0, 0.05, n_samples),
        })
        
        return df

    def test_init(self):
        """Test detector initialization."""
        detector = TrafficAnomalyDetector(contamination=0.05)
        
        assert detector.contamination == 0.05
        assert detector.random_state == 42
        assert detector.model is None

    def test_fit_predict_dataframe(self, sample_data):
        """Test fit and predict with DataFrame."""
        detector = TrafficAnomalyDetector(contamination=0.01)
        
        predictions = detector.fit_predict(sample_data)
        
        assert len(predictions) == len(sample_data)
        assert set(predictions).issubset({-1, 1})
        
        # Should detect approximately 1% as anomalies
        anomaly_rate = (predictions == -1).sum() / len(predictions)
        assert 0.005 < anomaly_rate < 0.02  # Allow some variance

    def test_fit_predict_array(self, sample_data):
        """Test fit and predict with numpy array."""
        detector = TrafficAnomalyDetector(contamination=0.01)
        
        X = sample_data.values
        predictions = detector.fit_predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

    def test_decision_function(self, sample_data):
        """Test decision function for anomaly scores."""
        detector = TrafficAnomalyDetector(contamination=0.01)
        detector.fit(sample_data)
        
        scores = detector.decision_function(sample_data)
        
        assert len(scores) == len(sample_data)
        assert isinstance(scores, np.ndarray)
        
        # Lower scores should be anomalies
        predictions = detector.predict(sample_data)
        anomaly_mask = predictions == -1
        
        assert scores[anomaly_mask].mean() < scores[~anomaly_mask].mean()

    def test_fit_before_predict(self):
        """Test that predict fails without fitting first."""
        detector = TrafficAnomalyDetector()
        X = np.random.normal(0, 1, (100, 3))
        
        with pytest.raises(ValueError, match="Model not trained"):
            detector.predict(X)

    def test_get_anomaly_info(self, sample_data):
        """Test anomaly information extraction."""
        detector = TrafficAnomalyDetector(contamination=0.01)
        
        predictions = detector.fit_predict(sample_data)
        
        # Test with predictions array
        info = detector.get_anomaly_info(sample_data, predictions)
        
        assert isinstance(info, dict)
        assert len(info) >= 0  # Should return dict of timestamp -> value
        
        # Test with column
        sample_data['is_anomaly_ml'] = (predictions == -1).astype(int)
        info2 = detector.get_anomaly_info(sample_data, anomaly_col='is_anomaly_ml')
        
        assert isinstance(info2, dict)

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random_state."""
        detector1 = TrafficAnomalyDetector(random_state=42)
        detector2 = TrafficAnomalyDetector(random_state=42)
        
        pred1 = detector1.fit_predict(sample_data)
        pred2 = detector2.fit_predict(sample_data)
        
        np.testing.assert_array_equal(pred1, pred2)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
def test_detect_anomalies_multimethod():
    """Test multi-method anomaly detection."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'request_count': np.random.normal(100, 20, 100),
        'unique_hosts': np.random.normal(50, 10, 100),
        'is_spike': 0,  # Z-score based
    })
    
    # Add some spikes
    df.loc[10:15, 'is_spike'] = 1
    df.loc[10:15, 'request_count'] = 300
    
    feature_cols = ['request_count', 'unique_hosts']
    result = detect_anomalies_multimethod(df, feature_cols, contamination=0.05)
    
    assert 'is_anomaly_ml' in result.columns
    assert 'anomaly_score_ml' in result.columns
    assert 'anomaly_agreement' in result.columns
    
    # Check agreement column
    assert result['anomaly_agreement'].sum() >= 0
