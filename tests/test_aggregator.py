"""Unit tests for the TimeAggregator module."""

from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.aggregator import TimeAggregator, AggregationConfig


class TestAggregationConfig:
    """Tests for AggregationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AggregationConfig(window="5min")

        assert config.window == "5min"
        assert config.fill_method == "zero"
        assert config.include_host_stats is True
        assert config.include_error_stats is True
        assert config.include_bytes_stats is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AggregationConfig(
            window="15min",
            fill_method="ffill",
            include_host_stats=False,
            include_error_stats=False,
            include_bytes_stats=False,
        )

        assert config.window == "15min"
        assert config.fill_method == "ffill"
        assert config.include_host_stats is False
        assert config.include_error_stats is False
        assert config.include_bytes_stats is False


class TestTimeAggregator:
    """Tests for TimeAggregator class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        timestamps = pd.date_range("1995-07-01 00:00:00", periods=100, freq="30s")
        return pd.DataFrame({
            "timestamp": timestamps,
            "host": [f"host{i % 10}.com" for i in range(100)],
            "method": ["GET"] * 80 + ["POST"] * 20,
            "url": ["/page.html"] * 100,
            "status": [200] * 90 + [404] * 10,
            "bytes": np.random.randint(100, 10000, 100),
            "is_error": [False] * 90 + [True] * 10,
            "is_success": [True] * 90 + [False] * 10,
        })

    @pytest.fixture
    def aggregator(self):
        """Create aggregator with 5-minute window."""
        return TimeAggregator(AggregationConfig(window="5min"))

    def test_initialization_defaults(self):
        """Test default initialization."""
        aggregator = TimeAggregator()
        assert aggregator.config.window == "5min"
        assert aggregator.config.fill_method == "zero"

    def test_initialization_custom_config(self):
        """Test custom configuration initialization."""
        config = AggregationConfig(window="1min", fill_method="ffill")
        aggregator = TimeAggregator(config)

        assert aggregator.config.window == "1min"
        assert aggregator.config.fill_method == "ffill"

    def test_aggregate_basic(self, aggregator, sample_df):
        """Test basic aggregation."""
        result = aggregator.aggregate(sample_df)

        assert "timestamp" in result.columns
        assert "request_count" in result.columns
        assert len(result) > 0

    def test_aggregate_request_count(self, aggregator, sample_df):
        """Test request count aggregation."""
        result = aggregator.aggregate(sample_df)

        # Total requests should be preserved
        assert result["request_count"].sum() == len(sample_df)

    def test_aggregate_unique_hosts(self, aggregator, sample_df):
        """Test unique hosts aggregation."""
        result = aggregator.aggregate(sample_df)

        assert "unique_hosts" in result.columns
        # Each window should have some unique hosts
        assert (result["unique_hosts"] > 0).any()

    def test_aggregate_error_stats(self, aggregator, sample_df):
        """Test error statistics aggregation."""
        result = aggregator.aggregate(sample_df)

        assert "error_count" in result.columns
        assert "error_rate" in result.columns

        # Total errors should be 10
        assert result["error_count"].sum() == 10

    def test_aggregate_bytes_stats(self, aggregator, sample_df):
        """Test bytes statistics aggregation."""
        result = aggregator.aggregate(sample_df)

        assert "bytes_total" in result.columns
        assert "bytes_avg" in result.columns
        assert "bytes_max" in result.columns

    def test_aggregate_derived_metrics(self, aggregator, sample_df):
        """Test derived metrics calculation."""
        result = aggregator.aggregate(sample_df)

        assert "requests_per_host" in result.columns
        assert "bytes_per_request" in result.columns

    def test_fill_missing_zero(self, sample_df):
        """Test zero fill for missing periods."""
        config = AggregationConfig(window="1min", fill_method="zero")
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Should have continuous time index
        time_diffs = result["timestamp"].diff().dropna()
        assert (time_diffs == pd.Timedelta("1min")).all()

    def test_fill_missing_ffill(self, sample_df):
        """Test forward fill for missing periods."""
        config = AggregationConfig(window="1min", fill_method="ffill")
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Should have no NaN values
        assert not result.isna().any().any()

    def test_fill_missing_interpolate(self, sample_df):
        """Test interpolation for missing periods."""
        config = AggregationConfig(window="1min", fill_method="interpolate")
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Should have no NaN values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not result[col].isna().any()

    def test_1minute_window(self, sample_df):
        """Test 1-minute aggregation window."""
        config = AggregationConfig(window="1min")
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Time diffs should be 1 minute
        time_diffs = result["timestamp"].diff().dropna()
        assert (time_diffs == pd.Timedelta("1min")).all()

    def test_5minute_window(self, sample_df):
        """Test 5-minute aggregation window."""
        config = AggregationConfig(window="5min")
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Time diffs should be 5 minutes
        time_diffs = result["timestamp"].diff().dropna()
        assert (time_diffs == pd.Timedelta("5min")).all()

    def test_15minute_window(self, sample_df):
        """Test 15-minute aggregation window."""
        config = AggregationConfig(window="15min")
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Time diffs should be 15 minutes
        time_diffs = result["timestamp"].diff().dropna()
        assert (time_diffs == pd.Timedelta("15min")).all()

    def test_exclude_host_stats(self, sample_df):
        """Test aggregation without host statistics."""
        config = AggregationConfig(window="5min", include_host_stats=False)
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Should still have request count but not unique hosts
        assert "request_count" in result.columns

    def test_exclude_error_stats(self, sample_df):
        """Test aggregation without error statistics."""
        config = AggregationConfig(window="5min", include_error_stats=False)
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Should not have error columns
        assert "error_count" not in result.columns
        assert "error_rate" not in result.columns

    def test_exclude_bytes_stats(self, sample_df):
        """Test aggregation without bytes statistics."""
        config = AggregationConfig(window="5min", include_bytes_stats=False)
        aggregator = TimeAggregator(config)

        result = aggregator.aggregate(sample_df)

        # Should not have bytes columns
        assert "bytes_total" not in result.columns
        assert "bytes_avg" not in result.columns

    def test_aggregate_multiple_windows(self, sample_df):
        """Test aggregation to multiple windows."""
        aggregator = TimeAggregator()

        results = aggregator.aggregate_multiple_windows(sample_df, windows=["1min", "5min"])

        assert "1min" in results
        assert "5min" in results
        assert len(results["1min"]) > len(results["5min"])

    def test_aggregate_multiple_windows_default(self, sample_df):
        """Test aggregation with default windows."""
        aggregator = TimeAggregator()

        results = aggregator.aggregate_multiple_windows(sample_df)

        assert "1min" in results
        assert "5min" in results
        assert "15min" in results

    def test_requests_per_host_calculation(self, aggregator, sample_df):
        """Test requests per host calculation."""
        result = aggregator.aggregate(sample_df)

        # requests_per_host = request_count / unique_hosts
        for _, row in result.iterrows():
            if row["unique_hosts"] > 0:
                expected = row["request_count"] / row["unique_hosts"]
                assert abs(row["requests_per_host"] - expected) < 0.001

    def test_bytes_per_request_calculation(self, aggregator, sample_df):
        """Test bytes per request calculation."""
        result = aggregator.aggregate(sample_df)

        # bytes_per_request = bytes_total / request_count
        for _, row in result.iterrows():
            if row["request_count"] > 0:
                expected = row["bytes_total"] / row["request_count"]
                assert abs(row["bytes_per_request"] - expected) < 0.001

    def test_count_columns_are_integers(self, aggregator, sample_df):
        """Test that count columns are integers."""
        result = aggregator.aggregate(sample_df)

        assert result["request_count"].dtype in [np.int32, np.int64]

    def test_empty_dataframe(self, aggregator):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([]),
            "host": pd.Series([], dtype="str"),
            "bytes": pd.Series([], dtype="float64"),
            "is_error": pd.Series([], dtype="bool"),
            "is_success": pd.Series([], dtype="bool"),
        })

        # Empty DataFrame aggregation may raise or return empty
        try:
            result = aggregator.aggregate(df)
            assert len(result) == 0
        except (ValueError, KeyError):
            pass  # Some pandas versions raise on empty resample

    def test_dataframe_with_index_timestamp(self, aggregator):
        """Test aggregation when timestamp is already the index."""
        timestamps = pd.date_range("1995-07-01 00:00:00", periods=50, freq="30s")
        df = pd.DataFrame({
            "host": ["host1.com"] * 50,
            "bytes": [100] * 50,
            "is_error": [False] * 50,
            "is_success": [True] * 50,
        }, index=timestamps)

        result = aggregator.aggregate(df)

        assert "request_count" in result.columns


class TestTimeAggregatorSaveLoad:
    """Tests for save and load functionality."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        timestamps = pd.date_range("1995-07-01 00:00:00", periods=50, freq="1min")
        return pd.DataFrame({
            "timestamp": timestamps,
            "host": ["host1.com"] * 50,
            "bytes": [100] * 50,
            "is_error": [False] * 50,
            "is_success": [True] * 50,
        })

    def test_save_parquet(self, sample_df):
        """Test saving to parquet format."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.parquet"
            aggregator.save(result, filepath, format="parquet")

            assert filepath.exists()

    def test_save_csv(self, sample_df):
        """Test saving to CSV format."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            aggregator.save(result, filepath, format="csv")

            assert filepath.exists()

    def test_save_invalid_format(self, sample_df):
        """Test saving with invalid format raises error."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            with pytest.raises(ValueError, match="Unsupported format"):
                aggregator.save(result, filepath, format="txt")

    def test_load_parquet(self, sample_df):
        """Test loading from parquet format."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.parquet"
            aggregator.save(result, filepath, format="parquet")

            loaded = TimeAggregator.load(filepath)

            assert len(loaded) == len(result)

    def test_load_csv(self, sample_df):
        """Test loading from CSV format."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            aggregator.save(result, filepath, format="csv")

            loaded = TimeAggregator.load(filepath)

            assert len(loaded) == len(result)

    def test_load_invalid_format(self):
        """Test loading with invalid format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.touch()

            with pytest.raises(ValueError, match="Unsupported format"):
                TimeAggregator.load(filepath)


class TestTimeAggregatorStats:
    """Tests for statistics functionality."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        timestamps = pd.date_range("1995-07-01 00:00:00", periods=100, freq="30s")
        return pd.DataFrame({
            "timestamp": timestamps,
            "host": ["host1.com"] * 100,
            "bytes": [100] * 100,
            "is_error": [False] * 100,
            "is_success": [True] * 100,
        })

    def test_get_stats(self, sample_df):
        """Test statistics generation."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        stats = aggregator.get_stats(result)

        assert "rows" in stats
        assert "window" in stats
        assert "date_range" in stats
        assert "request_stats" in stats
        assert "missing_periods" in stats

    def test_get_stats_request_stats(self, sample_df):
        """Test request statistics in stats output."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        stats = aggregator.get_stats(result)

        assert "total" in stats["request_stats"]
        assert "mean" in stats["request_stats"]
        assert "max" in stats["request_stats"]
        assert "std" in stats["request_stats"]

    def test_get_stats_total_requests(self, sample_df):
        """Test total request count in stats."""
        aggregator = TimeAggregator()
        result = aggregator.aggregate(sample_df)

        stats = aggregator.get_stats(result)

        # Total should equal original row count
        assert stats["request_stats"]["total"] == len(sample_df)


class TestColumnFlattening:
    """Tests for column flattening utility."""

    def test_flatten_multiindex(self):
        """Test flattening of MultiIndex columns."""
        aggregator = TimeAggregator()

        columns = pd.MultiIndex.from_tuples([
            ("host", "count"),
            ("host", "nunique"),
            ("bytes", "sum"),
        ])

        flattened = aggregator._flatten_columns(columns)

        assert flattened == ["host_count", "host_nunique", "bytes_sum"]

    def test_flatten_single_index(self):
        """Test handling of single index columns."""
        aggregator = TimeAggregator()

        columns = pd.Index(["col1", "col2", "col3"])

        flattened = aggregator._flatten_columns(columns)

        assert flattened == ["col1", "col2", "col3"]
