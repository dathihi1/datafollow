"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_log_data():
    """Create sample log data for testing."""
    n_records = 100
    timestamps = pd.date_range("1995-07-01 00:00:00", periods=n_records, freq="30s")

    return pd.DataFrame({
        "host": [f"host{i % 10}.example.com" for i in range(n_records)],
        "timestamp": timestamps,
        "method": ["GET"] * 80 + ["POST"] * 15 + ["HEAD"] * 5,
        "url": [f"/page{i % 20}.html" for i in range(n_records)],
        "status": [200] * 85 + [404] * 10 + [500] * 5,
        "bytes": np.random.randint(100, 10000, n_records),
    })


@pytest.fixture
def sample_cleaned_data(sample_log_data):
    """Create sample cleaned data with derived columns."""
    df = sample_log_data.copy()

    df["is_error"] = df["status"] >= 400
    df["is_success"] = df["status"] == 200
    df["status_category"] = pd.cut(
        df["status"],
        bins=[0, 199, 299, 399, 499, 599],
        labels=["1xx", "2xx", "3xx", "4xx", "5xx"],
    )

    return df


@pytest.fixture
def sample_aggregated_data():
    """Create sample aggregated time series data."""
    n_periods = 288  # 24 hours at 5-min intervals
    timestamps = pd.date_range("1995-07-01 00:00:00", periods=n_periods, freq="5min")

    # Create realistic daily pattern
    hours = np.arange(n_periods) * 5 / 60  # Hours from start
    hour_of_day = hours % 24

    # Base pattern: low at night, peak during day
    base_load = 50 + 100 * np.sin(np.pi * (hour_of_day - 6) / 12)
    base_load = np.maximum(base_load, 20)

    # Add noise
    noise = np.random.normal(0, 10, n_periods)
    request_count = (base_load + noise).astype(int)
    request_count = np.maximum(request_count, 1)

    return pd.DataFrame({
        "timestamp": timestamps,
        "request_count": request_count,
        "unique_hosts": (request_count * 0.3).astype(int),
        "error_count": (request_count * 0.05).astype(int),
        "error_rate": np.random.uniform(0.01, 0.1, n_periods),
        "bytes_total": request_count * np.random.randint(500, 2000, n_periods),
        "bytes_avg": np.random.uniform(500, 2000, n_periods),
    })


@pytest.fixture
def sample_load_series():
    """Create sample load time series for scaling tests."""
    n_periods = 100

    # Create load with spikes
    base_load = np.full(n_periods, 100)

    # Add some spikes
    spikes = np.zeros(n_periods)
    spike_indices = [20, 40, 60, 80]
    for idx in spike_indices:
        spikes[idx:idx + 5] = np.linspace(0, 100, 5)

    load = base_load + spikes + np.random.normal(0, 10, n_periods)
    return np.maximum(load, 10)


@pytest.fixture
def sample_features():
    """Create sample feature matrix for model testing."""
    n_samples = 500
    n_features = 20

    # Create random features
    X = np.random.randn(n_samples, n_features)

    # Create target with some relationship to features
    y = 50 + 10 * X[:, 0] + 5 * X[:, 1] + np.random.randn(n_samples) * 5

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")
