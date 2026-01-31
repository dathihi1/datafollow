"""Unit tests for the DataCleaner module."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from src.data.cleaner import DataCleaner, CleaningReport


class TestCleaningReport:
    """Tests for CleaningReport dataclass."""

    def test_creation(self):
        """Test report creation with all fields."""
        report = CleaningReport(
            original_rows=1000,
            final_rows=950,
            duplicates_removed=30,
            invalid_status_removed=10,
            negative_bytes_fixed=5,
            missing_periods=[(datetime(2023, 1, 1), datetime(2023, 1, 2))],
            outliers_detected=15,
        )

        assert report.original_rows == 1000
        assert report.final_rows == 950
        assert report.duplicates_removed == 30
        assert report.invalid_status_removed == 10
        assert report.negative_bytes_fixed == 5
        assert len(report.missing_periods) == 1
        assert report.outliers_detected == 15

    def test_str_representation(self):
        """Test string representation of report."""
        report = CleaningReport(
            original_rows=1000,
            final_rows=950,
            duplicates_removed=30,
            invalid_status_removed=10,
            negative_bytes_fixed=5,
            missing_periods=[],
            outliers_detected=15,
        )

        report_str = str(report)
        assert "1,000" in report_str
        assert "950" in report_str
        assert "30" in report_str


class TestDataCleaner:
    """Tests for DataCleaner class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "host": ["host1.com", "host2.com", "host1.com", "host3.com", "host1.com"],
            "timestamp": pd.to_datetime([
                "1995-07-01 00:00:00",
                "1995-07-01 00:01:00",
                "1995-07-01 00:00:00",  # Duplicate
                "1995-07-01 00:02:00",
                "1995-07-01 00:03:00",
            ]),
            "method": ["GET", "GET", "GET", "POST", "GET"],
            "url": ["/index.html", "/images/logo.gif", "/index.html", "/cgi-bin/form", "/shuttle/info.html"],
            "status": [200, 200, 200, 302, 404],
            "bytes": [1024, 5000, 1024, 0, -1],  # One negative bytes
        })

    @pytest.fixture
    def cleaner(self):
        """Create cleaner instance."""
        return DataCleaner()

    def test_initialization_defaults(self):
        """Test default initialization."""
        cleaner = DataCleaner()
        assert cleaner.remove_duplicates is True
        assert cleaner.validate_status is True

    def test_initialization_custom(self):
        """Test custom initialization."""
        cleaner = DataCleaner(remove_duplicates=False, validate_status=False)
        assert cleaner.remove_duplicates is False
        assert cleaner.validate_status is False

    def test_clean_removes_duplicates(self, cleaner, sample_df):
        """Test that duplicates are removed."""
        cleaned_df, report = cleaner.clean(sample_df)

        assert report.duplicates_removed == 1
        assert len(cleaned_df) == 4  # One duplicate removed

    def test_clean_no_duplicates_removed(self, sample_df):
        """Test with duplicate removal disabled."""
        cleaner = DataCleaner(remove_duplicates=False)
        cleaned_df, report = cleaner.clean(sample_df)

        assert report.duplicates_removed == 0
        assert len(cleaned_df) == 5

    def test_clean_fixes_negative_bytes(self, cleaner, sample_df):
        """Test that negative bytes are fixed."""
        cleaned_df, report = cleaner.clean(sample_df)

        assert report.negative_bytes_fixed == 1
        assert (cleaned_df["bytes"] >= 0).all()

    def test_clean_validates_status(self, cleaner, sample_df):
        """Test that status codes are validated."""
        cleaned_df, report = cleaner.clean(sample_df)

        assert "status_valid" in cleaned_df.columns
        # All our sample status codes are valid
        assert cleaned_df["status_valid"].all()

    def test_clean_adds_derived_columns(self, cleaner, sample_df):
        """Test that derived columns are added."""
        cleaned_df, report = cleaner.clean(sample_df)

        assert "status_category" in cleaned_df.columns
        assert "is_error" in cleaned_df.columns
        assert "is_success" in cleaned_df.columns
        assert "extension" in cleaned_df.columns
        assert "url_category" in cleaned_df.columns

    def test_status_category_assignment(self, cleaner, sample_df):
        """Test status category assignment."""
        cleaned_df, _ = cleaner.clean(sample_df)

        # Check specific status categories
        assert cleaned_df[cleaned_df["status"] == 200]["status_category"].iloc[0] == "2xx"
        assert cleaned_df[cleaned_df["status"] == 302]["status_category"].iloc[0] == "3xx"
        assert cleaned_df[cleaned_df["status"] == 404]["status_category"].iloc[0] == "4xx"

    def test_is_error_flag(self, cleaner, sample_df):
        """Test is_error flag assignment."""
        cleaned_df, _ = cleaner.clean(sample_df)

        # 404 should be marked as error
        assert cleaned_df[cleaned_df["status"] == 404]["is_error"].iloc[0] == True  # noqa: E712
        # 200 should not be marked as error
        assert cleaned_df[cleaned_df["status"] == 200]["is_error"].iloc[0] == False  # noqa: E712

    def test_is_success_flag(self, cleaner, sample_df):
        """Test is_success flag assignment."""
        cleaned_df, _ = cleaner.clean(sample_df)

        # 200 should be marked as success
        assert cleaned_df[cleaned_df["status"] == 200]["is_success"].iloc[0] == True  # noqa: E712
        # 302 should not be marked as success
        assert cleaned_df[cleaned_df["status"] == 302]["is_success"].iloc[0] == False  # noqa: E712

    def test_extension_extraction(self, cleaner, sample_df):
        """Test file extension extraction."""
        cleaned_df, _ = cleaner.clean(sample_df)

        # Check extensions are extracted
        extensions = cleaned_df["extension"].tolist()
        assert "html" in extensions
        assert "gif" in extensions

    def test_url_categorization(self, cleaner, sample_df):
        """Test URL categorization."""
        cleaned_df, _ = cleaner.clean(sample_df)

        # Check URL categories
        categories = cleaned_df["url_category"].tolist()
        assert "images" in categories
        assert "shuttle" in categories
        assert "cgi" in categories

    def test_sort_by_timestamp(self, cleaner, sample_df):
        """Test that DataFrame is sorted by timestamp."""
        cleaned_df, _ = cleaner.clean(sample_df)

        timestamps = cleaned_df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_detect_missing_periods(self, cleaner):
        """Test missing period detection."""
        # Create DataFrame with a large gap
        df = pd.DataFrame({
            "host": ["host1.com", "host2.com"],
            "timestamp": pd.to_datetime([
                "1995-07-01 00:00:00",
                "1995-07-01 03:00:00",  # 3 hour gap
            ]),
            "method": ["GET", "GET"],
            "url": ["/index.html", "/page.html"],
            "status": [200, 200],
            "bytes": [100, 200],
        })

        _, report = cleaner.clean(df)

        # Should detect the gap
        assert len(report.missing_periods) > 0

    def test_detect_no_missing_periods(self, cleaner, sample_df):
        """Test when there are no missing periods."""
        _, report = cleaner.clean(sample_df)

        # Small gaps shouldn't be reported
        assert len(report.missing_periods) == 0

    def test_outlier_detection(self, cleaner):
        """Test outlier detection in bytes."""
        # Create DataFrame with an outlier
        df = pd.DataFrame({
            "host": ["host1.com"] * 100,
            "timestamp": pd.date_range("1995-07-01", periods=100, freq="1min"),
            "method": ["GET"] * 100,
            "url": ["/page.html"] * 100,
            "status": [200] * 100,
            "bytes": [100] * 99 + [1000000],  # Last one is an outlier
        })

        cleaned_df, report = cleaner.clean(df)

        assert report.outliers_detected >= 1
        assert "bytes_outlier" in cleaned_df.columns

    def test_timezone_handling(self, cleaner):
        """Test that timezone-aware timestamps are handled."""
        df = pd.DataFrame({
            "host": ["host1.com"],
            "timestamp": pd.to_datetime(["1995-07-01 00:00:00+00:00"]),
            "method": ["GET"],
            "url": ["/index.html"],
            "status": [200],
            "bytes": [100],
        })

        cleaned_df, _ = cleaner.clean(df)

        # Should be timezone-naive
        assert cleaned_df["timestamp"].dt.tz is None

    def test_invalid_status_detection(self, cleaner):
        """Test detection of invalid status codes."""
        df = pd.DataFrame({
            "host": ["host1.com", "host2.com"],
            "timestamp": pd.to_datetime(["1995-07-01 00:00:00", "1995-07-01 00:01:00"]),
            "method": ["GET", "GET"],
            "url": ["/page1.html", "/page2.html"],
            "status": [200, 999],  # 999 is invalid
            "bytes": [100, 200],
        })

        cleaned_df, report = cleaner.clean(df)

        assert report.invalid_status_removed == 1
        assert cleaned_df[cleaned_df["status"] == 999]["status_valid"].iloc[0] == False  # noqa: E712

    def test_empty_dataframe(self, cleaner):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "host": pd.Series([], dtype="str"),
            "timestamp": pd.to_datetime([]),
            "method": pd.Series([], dtype="str"),
            "url": pd.Series([], dtype="str"),
            "status": pd.Series([], dtype="int64"),
            "bytes": pd.Series([], dtype="int64"),
        })

        cleaned_df, report = cleaner.clean(df)

        assert len(cleaned_df) == 0
        assert report.original_rows == 0
        assert report.final_rows == 0


class TestDataCleanerSaveLoad:
    """Tests for save and load functionality."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "host": ["host1.com", "host2.com"],
            "timestamp": pd.to_datetime(["1995-07-01 00:00:00", "1995-07-01 00:01:00"]),
            "method": ["GET", "POST"],
            "url": ["/index.html", "/form"],
            "status": [200, 302],
            "bytes": [1024, 512],
        })

    def test_save_parquet(self, sample_df):
        """Test saving to parquet format."""
        cleaner = DataCleaner()
        cleaned_df, _ = cleaner.clean(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.parquet"
            cleaner.save(cleaned_df, filepath, format="parquet")

            assert filepath.exists()

    def test_save_csv(self, sample_df):
        """Test saving to CSV format."""
        cleaner = DataCleaner()
        cleaned_df, _ = cleaner.clean(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            cleaner.save(cleaned_df, filepath, format="csv")

            assert filepath.exists()

    def test_save_invalid_format(self, sample_df):
        """Test saving with invalid format raises error."""
        cleaner = DataCleaner()
        cleaned_df, _ = cleaner.clean(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            with pytest.raises(ValueError, match="Unsupported format"):
                cleaner.save(cleaned_df, filepath, format="txt")

    def test_load_parquet(self, sample_df):
        """Test loading from parquet format."""
        cleaner = DataCleaner()
        cleaned_df, _ = cleaner.clean(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.parquet"
            cleaner.save(cleaned_df, filepath, format="parquet")

            loaded_df = DataCleaner.load(filepath)

            assert len(loaded_df) == len(cleaned_df)

    def test_load_csv(self, sample_df):
        """Test loading from CSV format."""
        cleaner = DataCleaner()
        cleaned_df, _ = cleaner.clean(sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            cleaner.save(cleaned_df, filepath, format="csv")

            loaded_df = DataCleaner.load(filepath)

            assert len(loaded_df) == len(cleaned_df)

    def test_load_invalid_format(self):
        """Test loading with invalid format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.touch()

            with pytest.raises(ValueError, match="Unsupported format"):
                DataCleaner.load(filepath)


class TestUrlCategorization:
    """Tests for URL categorization logic."""

    @pytest.fixture
    def cleaner(self):
        """Create cleaner instance."""
        return DataCleaner()

    def test_images_category(self, cleaner):
        """Test images URL category."""
        urls = pd.Series(["/images/logo.gif", "/images/nasa.jpg"])
        categories = cleaner._categorize_url(urls)
        assert (categories == "images").all()

    def test_shuttle_category(self, cleaner):
        """Test shuttle URL category."""
        urls = pd.Series(["/shuttle/mission.html", "/shuttle/countdown/"])
        categories = cleaner._categorize_url(urls)
        assert (categories == "shuttle").all()

    def test_history_category(self, cleaner):
        """Test history URL category."""
        urls = pd.Series(["/history/apollo/", "/history/mercury.html"])
        categories = cleaner._categorize_url(urls)
        assert (categories == "history").all()

    def test_cgi_category(self, cleaner):
        """Test CGI URL category."""
        urls = pd.Series(["/cgi-bin/form.cgi", "/cgi-bin/search"])
        categories = cleaner._categorize_url(urls)
        assert (categories == "cgi").all()

    def test_homepage_category(self, cleaner):
        """Test homepage URL category."""
        urls = pd.Series(["/", "/ksc.html"])
        categories = cleaner._categorize_url(urls)
        assert (categories == "homepage").all()

    def test_other_category(self, cleaner):
        """Test other URL category."""
        urls = pd.Series(["/unknown/path", "/random.html"])
        categories = cleaner._categorize_url(urls)
        assert (categories == "other").all()
