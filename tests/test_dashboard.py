"""Unit tests for the Streamlit dashboard."""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Mock streamlit before importing dashboard module
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.dashboard import (
    get_config,
    generate_sample_load,
    run_simulation,
    _load_csv_data,
    _load_txt_data,
    DAILY_PERIODS_5MIN,
    MIN_LOAD_VALUE,
    MAX_CSV_SIZE_MB,
    MAX_CSV_ROWS,
)
from src.scaling.config import ScalingConfig, BALANCED_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_balanced_config(self):
        """Test getting balanced configuration."""
        config = get_config("balanced")
        assert config == BALANCED_CONFIG

    def test_get_conservative_config(self):
        """Test getting conservative configuration."""
        config = get_config("conservative")
        assert config == CONSERVATIVE_CONFIG

    def test_get_aggressive_config(self):
        """Test getting aggressive configuration."""
        config = get_config("aggressive")
        assert config == AGGRESSIVE_CONFIG

    def test_get_invalid_config_returns_balanced(self):
        """Test that invalid preset returns balanced config."""
        config = get_config("invalid")
        assert config == BALANCED_CONFIG


class TestGenerateSampleLoad:
    """Tests for generate_sample_load function."""

    def test_default_periods(self):
        """Test generation with default periods."""
        loads = generate_sample_load()
        assert len(loads) == DAILY_PERIODS_5MIN

    def test_custom_periods(self):
        """Test generation with custom periods."""
        loads = generate_sample_load(n_periods=100)
        assert len(loads) == 100

    def test_minimum_value(self):
        """Test that all values are at least MIN_LOAD_VALUE."""
        loads = generate_sample_load(n_periods=500, seed=123)
        assert np.all(loads >= MIN_LOAD_VALUE)

    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        loads1 = generate_sample_load(seed=42)
        loads2 = generate_sample_load(seed=42)
        np.testing.assert_array_equal(loads1, loads2)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        loads1 = generate_sample_load(seed=1)
        loads2 = generate_sample_load(seed=2)
        assert not np.array_equal(loads1, loads2)

    def test_realistic_range(self):
        """Test that loads are in realistic range."""
        loads = generate_sample_load(n_periods=500)
        assert np.mean(loads) > 30  # Has some significant load
        assert np.max(loads) < 500  # Not unreasonably high


class TestRunSimulation:
    """Tests for run_simulation function."""

    @pytest.fixture
    def sample_loads(self):
        """Create sample load data."""
        return np.array([100, 150, 200, 250, 300, 250, 200, 150, 100])

    def test_simulation_returns_tuple(self, sample_loads):
        """Test that simulation returns expected tuple."""
        result = run_simulation(sample_loads, BALANCED_CONFIG, "balanced")

        assert result is not None
        assert len(result) == 4
        metrics, servers, utilizations, costs = result
        assert isinstance(metrics, dict)
        assert isinstance(servers, list)
        assert isinstance(utilizations, list)
        assert isinstance(costs, list)

    def test_simulation_reactive_policy(self, sample_loads):
        """Test simulation with reactive policy."""
        result = run_simulation(sample_loads, BALANCED_CONFIG, "reactive")
        assert result is not None

    def test_simulation_predictive_policy(self, sample_loads):
        """Test simulation with predictive policy."""
        result = run_simulation(sample_loads, BALANCED_CONFIG, "predictive")
        assert result is not None

    def test_simulation_metrics_keys(self, sample_loads):
        """Test that metrics contain expected keys."""
        result = run_simulation(sample_loads, BALANCED_CONFIG, "balanced")
        metrics = result[0]

        expected_keys = [
            'total_cost',
            'avg_servers',
            'avg_utilization',
            'sla_violations',
            'scaling_events',
            'min_servers',
            'max_servers',
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_simulation_list_lengths_match(self, sample_loads):
        """Test that returned lists match input length."""
        result = run_simulation(sample_loads, BALANCED_CONFIG, "balanced")
        _, servers, utilizations, costs = result

        assert len(servers) == len(sample_loads)
        assert len(utilizations) == len(sample_loads)
        assert len(costs) == len(sample_loads)


class TestLoadCSVData:
    """Tests for _load_csv_data function."""

    def test_load_valid_csv_with_load_column(self):
        """Test loading valid CSV with 'load' column."""
        csv_content = "timestamp,load\n2023-01-01,100\n2023-01-02,150\n2023-01-03,200"
        mock_file = io.BytesIO(csv_content.encode('utf-8'))
        mock_file.size = len(csv_content)

        loads = _load_csv_data(mock_file)

        assert loads is not None
        assert len(loads) == 3
        np.testing.assert_array_equal(loads, [100, 150, 200])

    def test_load_valid_csv_with_request_count_column(self):
        """Test loading valid CSV with 'request_count' column."""
        csv_content = "timestamp,request_count\n2023-01-01,50\n2023-01-02,75"
        mock_file = io.BytesIO(csv_content.encode('utf-8'))
        mock_file.size = len(csv_content)

        loads = _load_csv_data(mock_file)

        assert loads is not None
        assert len(loads) == 2
        np.testing.assert_array_equal(loads, [50, 75])

    def test_load_csv_missing_column(self):
        """Test loading CSV without required column."""
        csv_content = "timestamp,value\n2023-01-01,100"
        mock_file = io.BytesIO(csv_content.encode('utf-8'))
        mock_file.size = len(csv_content)

        loads = _load_csv_data(mock_file)

        assert loads is None

    def test_load_csv_with_nan_values(self):
        """Test loading CSV with NaN values."""
        csv_content = "load\n100\n\n200"
        mock_file = io.BytesIO(csv_content.encode('utf-8'))
        mock_file.size = len(csv_content)

        loads = _load_csv_data(mock_file)

        assert loads is not None
        # NaN should be replaced with 0
        assert not np.any(np.isnan(loads))

    def test_load_csv_with_negative_values(self):
        """Test loading CSV with negative values."""
        csv_content = "load\n100\n-50\n200"
        mock_file = io.BytesIO(csv_content.encode('utf-8'))
        mock_file.size = len(csv_content)

        loads = _load_csv_data(mock_file)

        assert loads is not None
        # Negative values should be set to 0
        assert np.all(loads >= 0)

    def test_load_csv_file_too_large(self):
        """Test loading CSV file that exceeds size limit."""
        csv_content = "load\n100"
        mock_file = io.BytesIO(csv_content.encode('utf-8'))
        mock_file.size = (MAX_CSV_SIZE_MB + 1) * 1024 * 1024  # Exceed max size

        loads = _load_csv_data(mock_file)

        assert loads is None


class TestLoadTXTData:
    """Tests for _load_txt_data function."""

    def test_load_valid_txt_one_per_line(self):
        """Test loading TXT with one number per line."""
        txt_content = "100\n150\n200\n250"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        assert len(loads) == 4
        np.testing.assert_array_equal(loads, [100, 150, 200, 250])

    def test_load_valid_txt_space_separated(self):
        """Test loading TXT with space-separated values."""
        txt_content = "100 150 200"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        assert len(loads) == 3
        np.testing.assert_array_equal(loads, [100, 150, 200])

    def test_load_valid_txt_comma_separated(self):
        """Test loading TXT with comma-separated values."""
        txt_content = "100,150,200,250"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        assert len(loads) == 4

    def test_load_txt_with_comments(self):
        """Test loading TXT with comment lines."""
        txt_content = "# This is a comment\n100\n# Another comment\n200"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        assert len(loads) == 2
        np.testing.assert_array_equal(loads, [100, 200])

    def test_load_txt_with_empty_lines(self):
        """Test loading TXT with empty lines."""
        txt_content = "100\n\n200\n\n300"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        assert len(loads) == 3

    def test_load_txt_with_nan_values(self):
        """Test loading TXT with NaN values."""
        txt_content = "100\nnan\n200"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        # NaN should be replaced with 0
        assert not np.any(np.isnan(loads))

    def test_load_txt_with_negative_values(self):
        """Test loading TXT with negative values."""
        txt_content = "100\n-50\n200"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        assert np.all(loads >= 0)

    def test_load_txt_no_valid_numbers(self):
        """Test loading TXT with no valid numbers."""
        txt_content = "abc\ndef\nghi"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is None

    def test_load_txt_file_too_large(self):
        """Test loading TXT file that exceeds size limit."""
        txt_content = "100"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = (MAX_CSV_SIZE_MB + 1) * 1024 * 1024

        loads = _load_txt_data(mock_file)

        assert loads is None

    def test_load_txt_mixed_format(self):
        """Test loading TXT with mixed separators."""
        txt_content = "100\n150 200\n250,300"
        mock_file = io.BytesIO(txt_content.encode('utf-8'))
        mock_file.size = len(txt_content)

        loads = _load_txt_data(mock_file)

        assert loads is not None
        assert len(loads) == 5


class TestDashboardIntegration:
    """Integration tests for dashboard components."""

    def test_full_simulation_pipeline(self):
        """Test complete simulation pipeline."""
        # Generate sample data
        loads = generate_sample_load(n_periods=100, seed=42)

        # Get config
        config = get_config("balanced")

        # Run simulation
        result = run_simulation(loads, config, "predictive")

        assert result is not None
        metrics, servers, utilizations, costs = result

        # Verify metrics
        assert metrics['total_cost'] >= 0
        assert 0 <= metrics['avg_utilization'] <= 2.0  # Can be > 1 if overloaded
        assert metrics['min_servers'] >= config.min_servers
        assert metrics['max_servers'] <= config.max_servers

        # Verify time series
        assert len(servers) == len(loads)
        assert len(costs) == len(loads)
        assert all(s >= config.min_servers for s in servers)
        assert all(c >= 0 for c in costs)

    def test_all_config_presets(self):
        """Test all configuration presets work."""
        loads = generate_sample_load(n_periods=50, seed=42)

        for preset in ["balanced", "conservative", "aggressive"]:
            config = get_config(preset)
            result = run_simulation(loads, config, "balanced")
            assert result is not None, f"Simulation failed for preset: {preset}"

    def test_all_policy_types(self):
        """Test all policy types work."""
        loads = generate_sample_load(n_periods=50, seed=42)
        config = get_config("balanced")

        for policy in ["balanced", "reactive", "predictive"]:
            result = run_simulation(loads, config, policy)
            assert result is not None, f"Simulation failed for policy: {policy}"

    def test_edge_case_low_load(self):
        """Test simulation with very low load."""
        loads = np.full(20, MIN_LOAD_VALUE)
        config = get_config("balanced")

        result = run_simulation(loads, config, "balanced")

        assert result is not None
        metrics = result[0]
        # Should use minimum servers
        assert metrics['min_servers'] == config.min_servers

    def test_edge_case_high_load(self):
        """Test simulation with very high load."""
        config = get_config("balanced")
        # Load that would require max servers
        high_load = config.max_servers * config.requests_per_server * 2
        loads = np.full(20, high_load)

        result = run_simulation(loads, config, "balanced")

        assert result is not None
        metrics = result[0]
        # Should hit max servers
        assert metrics['max_servers'] == config.max_servers

    def test_edge_case_spike_load(self):
        """Test simulation with sudden spike."""
        base_load = np.full(50, 100)
        # Add spike in the middle
        base_load[20:30] = 1000

        config = get_config("balanced")
        result = run_simulation(base_load, config, "reactive")

        assert result is not None
        metrics = result[0]
        # Should have scaling events due to spike
        assert metrics['scaling_events'] > 0


class TestConstants:
    """Tests for dashboard constants."""

    def test_daily_periods_value(self):
        """Test DAILY_PERIODS_5MIN is correct for 24 hours."""
        # 24 hours * 12 periods per hour (5-min intervals) = 288
        assert DAILY_PERIODS_5MIN == 288

    def test_min_load_value_positive(self):
        """Test MIN_LOAD_VALUE is positive."""
        assert MIN_LOAD_VALUE > 0

    def test_max_csv_size_reasonable(self):
        """Test MAX_CSV_SIZE_MB is reasonable."""
        assert MAX_CSV_SIZE_MB >= 100  # At least 100MB
        assert MAX_CSV_SIZE_MB <= 1000  # Not more than 1GB

    def test_max_csv_rows_reasonable(self):
        """Test MAX_CSV_ROWS is reasonable."""
        assert MAX_CSV_ROWS >= 100_000
        assert MAX_CSV_ROWS <= 100_000_000
