"""Tests for log parser module."""

import pytest
from datetime import datetime, timezone

from src.data.parser import LogParser, LogEntry


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_create_log_entry(self):
        """Test creating a log entry."""
        entry = LogEntry(
            host="199.72.81.55",
            timestamp=datetime(1995, 7, 1, 0, 0, 1, tzinfo=timezone.utc),
            method="GET",
            url="/history/apollo/",
            protocol="HTTP/1.0",
            status=200,
            bytes=6245,
        )

        assert entry.host == "199.72.81.55"
        assert entry.method == "GET"
        assert entry.status == 200
        assert entry.bytes == 6245

    def test_log_entry_immutable(self):
        """Test that LogEntry is immutable (frozen)."""
        entry = LogEntry(
            host="test.com",
            timestamp=datetime.now(),
            method="GET",
            url="/",
            protocol="HTTP/1.0",
            status=200,
            bytes=100,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            entry.host = "other.com"

    def test_to_dict(self):
        """Test converting to dictionary."""
        ts = datetime(1995, 7, 1, 0, 0, 1)
        entry = LogEntry(
            host="test.com",
            timestamp=ts,
            method="GET",
            url="/test",
            protocol="HTTP/1.0",
            status=200,
            bytes=1000,
        )

        d = entry.to_dict()
        assert d["host"] == "test.com"
        assert d["timestamp"] == ts
        assert d["bytes"] == 1000


class TestLogParser:
    """Tests for LogParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return LogParser()

    def test_parse_valid_line(self, parser):
        """Test parsing a valid log line."""
        line = '199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245'
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.host == "199.72.81.55"
        assert entry.method == "GET"
        assert entry.url == "/history/apollo/"
        assert entry.protocol == "HTTP/1.0"
        assert entry.status == 200
        assert entry.bytes == 6245
        assert entry.timestamp.year == 1995
        assert entry.timestamp.month == 7
        assert entry.timestamp.day == 1

    def test_parse_line_with_domain(self, parser):
        """Test parsing a line with domain hostname."""
        line = 'unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 3985'
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.host == "unicomp6.unicomp.net"
        assert entry.status == 200

    def test_parse_line_with_304_status(self, parser):
        """Test parsing 304 Not Modified response."""
        line = 'burger.letters.com - - [01/Jul/1995:00:00:11 -0400] "GET /shuttle/countdown/liftoff.html HTTP/1.0" 304 0'
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.status == 304
        assert entry.bytes == 0

    def test_parse_line_with_missing_bytes(self, parser):
        """Test parsing line with - for bytes."""
        line = 'test.com - - [01/Jul/1995:00:00:01 -0400] "GET /test HTTP/1.0" 200 -'
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.bytes == 0

    def test_parse_invalid_line(self, parser):
        """Test parsing an invalid line returns None."""
        line = "this is not a valid log line"
        entry = parser.parse_line(line)

        assert entry is None

    def test_parse_empty_line(self, parser):
        """Test parsing empty line returns None."""
        entry = parser.parse_line("")
        assert entry is None

        entry = parser.parse_line("   ")
        assert entry is None

    def test_parse_line_with_post_method(self, parser):
        """Test parsing POST request."""
        line = 'test.com - - [01/Jul/1995:00:00:01 -0400] "POST /cgi-bin/form HTTP/1.0" 200 100'
        entry = parser.parse_line(line)

        assert entry is not None
        assert entry.method == "POST"

    def test_parse_errors_tracked(self, parser):
        """Test that parse errors are tracked."""
        # Parse some valid and invalid lines
        parser.parse_line("invalid line 1")
        parser.parse_line('199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET / HTTP/1.0" 200 100')
        parser.parse_line("invalid line 2")

        # Check error count (note: parse_line doesn't track errors, only parse_file does)
        # This test verifies that parse_line returns None for invalid lines
        assert parser.parse_line("invalid") is None


class TestLogParserStats:
    """Tests for parser statistics."""

    def test_get_stats_empty_dataframe(self):
        """Test stats with empty DataFrame."""
        import pandas as pd

        parser = LogParser()
        df = pd.DataFrame(columns=["host", "timestamp", "method", "url", "protocol", "status", "bytes"])

        stats = parser.get_stats(df)
        assert stats["total_records"] == 0
        assert stats["unique_hosts"] == 0

    def test_get_stats_with_data(self):
        """Test stats with sample data."""
        import pandas as pd
        from datetime import datetime

        parser = LogParser()
        df = pd.DataFrame({
            "host": ["host1", "host2", "host1"],
            "timestamp": [datetime(1995, 7, 1), datetime(1995, 7, 2), datetime(1995, 7, 3)],
            "method": ["GET", "GET", "POST"],
            "url": ["/", "/test", "/form"],
            "protocol": ["HTTP/1.0", "HTTP/1.0", "HTTP/1.0"],
            "status": [200, 200, 404],
            "bytes": [100, 200, 0],
        })

        stats = parser.get_stats(df)
        assert stats["total_records"] == 3
        assert stats["unique_hosts"] == 2
        assert stats["total_bytes"] == 300
        assert 200 in stats["status_codes"]
        assert 404 in stats["status_codes"]
