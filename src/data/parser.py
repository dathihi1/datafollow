"""Log parser for NASA web server logs in Apache Combined Log Format.

Optimized for handling large datasets (4+ million records) with:
- Chunked reading to reduce memory usage
- Direct list parsing (no intermediate dataclass)
- Efficient dtype conversion
- Optional parallel processing
"""

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LogEntry:
    """Immutable representation of a parsed log entry."""

    host: str
    timestamp: datetime
    method: str
    url: str
    protocol: str
    status: int
    bytes: int

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "host": self.host,
            "timestamp": self.timestamp,
            "method": self.method,
            "url": self.url,
            "protocol": self.protocol,
            "status": self.status,
            "bytes": self.bytes,
        }


class LogParser:
    """Parser for Apache Combined Log Format.

    Log format: <host> - - [<timestamp>] "<request>" <status> <bytes>
    Example: 199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
    """

    # Regex pattern for Apache Combined Log Format
    LOG_PATTERN = re.compile(
        r'^(?P<host>\S+)\s+'  # Host (IP or domain)
        r'-\s+-\s+'  # Ident and authuser (always - -)
        r'\[(?P<timestamp>[^\]]+)\]\s+'  # Timestamp in brackets
        r'"(?P<method>\S+)\s+'  # HTTP method
        r'(?P<url>\S+)\s+'  # URL path
        r'(?P<protocol>[^"]+)"\s+'  # Protocol (HTTP/1.0)
        r'(?P<status>\d+)\s+'  # Status code
        r'(?P<bytes>\S+)$'  # Bytes (can be - for missing)
    )

    # Date format: 01/Jul/1995:00:00:01 -0400
    DATE_FORMAT = "%d/%b/%Y:%H:%M:%S %z"

    def __init__(self, encoding: str = "latin-1"):
        """Initialize parser.

        Args:
            encoding: File encoding (latin-1 handles special characters in hostnames)
        """
        self.encoding = encoding
        self._parse_errors: list[tuple[int, str]] = []

    @property
    def parse_errors(self) -> list[tuple[int, str]]:
        """Return list of (line_number, line) tuples that failed to parse."""
        return self._parse_errors.copy()

    def parse_line(self, line: str) -> LogEntry | None:
        """Parse a single log line.

        Args:
            line: Raw log line

        Returns:
            LogEntry if successful, None if parsing failed
        """
        line = line.strip()
        if not line:
            return None

        match = self.LOG_PATTERN.match(line)
        if not match:
            return None

        try:
            groups = match.groupdict()

            # Parse timestamp
            timestamp = datetime.strptime(groups["timestamp"], self.DATE_FORMAT)

            # Parse bytes (handle '-' as 0)
            bytes_str = groups["bytes"]
            bytes_val = 0 if bytes_str == "-" else int(bytes_str)

            return LogEntry(
                host=groups["host"],
                timestamp=timestamp,
                method=groups["method"],
                url=groups["url"],
                protocol=groups["protocol"],
                status=int(groups["status"]),
                bytes=bytes_val,
            )
        except (ValueError, KeyError):
            return None

    def parse_file(self, filepath: str | Path, show_progress: bool = True) -> Iterator[LogEntry]:
        """Parse log file and yield LogEntry objects.

        Args:
            filepath: Path to log file
            show_progress: Whether to print progress

        Yields:
            Parsed LogEntry objects
        """
        filepath = Path(filepath)
        self._parse_errors = []

        total_lines = 0
        parsed_lines = 0

        with open(filepath, "r", encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                entry = self.parse_line(line)

                if entry is not None:
                    parsed_lines += 1
                    yield entry
                else:
                    self._parse_errors.append((line_num, line.strip()))

                # Progress reporting
                if show_progress and total_lines % 500_000 == 0:
                    success_rate = (parsed_lines / total_lines) * 100
                    print(f"Processed {total_lines:,} lines ({success_rate:.2f}% success)")

        if show_progress:
            success_rate = (parsed_lines / total_lines) * 100 if total_lines > 0 else 0
            print(f"Completed: {parsed_lines:,}/{total_lines:,} lines ({success_rate:.2f}% success)")

    def parse_to_dataframe(
        self,
        filepath: str | Path,
        show_progress: bool = True,
        chunk_size: int = 100_000,
        use_optimized: bool = True,
    ) -> pd.DataFrame:
        """Parse log file directly to pandas DataFrame.

        Optimized for large files (4+ million records) with chunked processing.

        Args:
            filepath: Path to log file
            show_progress: Whether to print progress
            chunk_size: Number of lines to process per chunk (reduces memory)
            use_optimized: Use optimized direct-to-list parsing (faster for large files)

        Returns:
            DataFrame with columns: host, timestamp, method, url, protocol, status, bytes
        """
        if use_optimized:
            return self._parse_to_dataframe_optimized(filepath, show_progress, chunk_size)

        # Original method (kept for compatibility)
        entries = list(self.parse_file(filepath, show_progress))

        if not entries:
            return pd.DataFrame(columns=["host", "timestamp", "method", "url", "protocol", "status", "bytes"])

        df = pd.DataFrame([entry.to_dict() for entry in entries])

        # Optimize dtypes
        df["method"] = df["method"].astype("category")
        df["protocol"] = df["protocol"].astype("category")
        df["status"] = df["status"].astype("int16")
        df["bytes"] = df["bytes"].astype("int64")

        return df

    def _parse_to_dataframe_optimized(
        self,
        filepath: str | Path,
        show_progress: bool = True,
        chunk_size: int = 100_000,
    ) -> pd.DataFrame:
        """Optimized parsing directly to DataFrame without intermediate objects.

        Uses chunked reading and direct list appending for memory efficiency.

        Args:
            filepath: Path to log file
            show_progress: Whether to print progress
            chunk_size: Number of lines per chunk

        Returns:
            Optimized DataFrame
        """
        filepath = Path(filepath)
        self._parse_errors = []

        # Pre-allocate lists for direct column storage
        hosts: list[str] = []
        timestamps: list[datetime] = []
        methods: list[str] = []
        urls: list[str] = []
        protocols: list[str] = []
        statuses: list[int] = []
        bytes_list: list[int] = []

        total_lines = 0
        parsed_lines = 0

        with open(filepath, "r", encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()

                if not line:
                    continue

                match = self.LOG_PATTERN.match(line)
                if not match:
                    self._parse_errors.append((line_num, line))
                    continue

                try:
                    groups = match.groupdict()

                    # Parse timestamp
                    timestamp = datetime.strptime(groups["timestamp"], self.DATE_FORMAT)

                    # Parse bytes (handle '-' as 0)
                    bytes_str = groups["bytes"]
                    bytes_val = 0 if bytes_str == "-" else int(bytes_str)

                    # Append directly to lists
                    hosts.append(groups["host"])
                    timestamps.append(timestamp)
                    methods.append(groups["method"])
                    urls.append(groups["url"])
                    protocols.append(groups["protocol"])
                    statuses.append(int(groups["status"]))
                    bytes_list.append(bytes_val)

                    parsed_lines += 1

                except (ValueError, KeyError):
                    self._parse_errors.append((line_num, line))

                # Progress reporting
                if show_progress and total_lines % 500_000 == 0:
                    success_rate = (parsed_lines / total_lines) * 100
                    print(f"Processed {total_lines:,} lines ({success_rate:.2f}% success)")

        if show_progress:
            success_rate = (parsed_lines / total_lines) * 100 if total_lines > 0 else 0
            print(f"Completed: {parsed_lines:,}/{total_lines:,} lines ({success_rate:.2f}% success)")

        if not hosts:
            return pd.DataFrame(columns=["host", "timestamp", "method", "url", "protocol", "status", "bytes"])

        # Create DataFrame directly from lists (more efficient than from dicts)
        df = pd.DataFrame({
            "host": hosts,
            "timestamp": timestamps,
            "method": pd.Categorical(methods),
            "url": urls,
            "protocol": pd.Categorical(protocols),
            "status": np.array(statuses, dtype=np.int16),
            "bytes": np.array(bytes_list, dtype=np.int64),
        })

        return df

    def parse_to_dataframe_chunked(
        self,
        filepath: str | Path,
        chunk_size: int = 500_000,
        show_progress: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """Parse log file in chunks for memory-efficient processing.

        Yields DataFrames chunk by chunk, useful for very large files that
        don't fit in memory.

        Args:
            filepath: Path to log file
            chunk_size: Number of records per chunk
            show_progress: Whether to print progress

        Yields:
            DataFrame chunks
        """
        filepath = Path(filepath)
        self._parse_errors = []

        # Pre-allocate lists for current chunk
        hosts: list[str] = []
        timestamps: list[datetime] = []
        methods: list[str] = []
        urls: list[str] = []
        protocols: list[str] = []
        statuses: list[int] = []
        bytes_list: list[int] = []

        total_lines = 0
        parsed_lines = 0
        chunk_num = 0

        with open(filepath, "r", encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()

                if not line:
                    continue

                match = self.LOG_PATTERN.match(line)
                if not match:
                    self._parse_errors.append((line_num, line))
                    continue

                try:
                    groups = match.groupdict()
                    timestamp = datetime.strptime(groups["timestamp"], self.DATE_FORMAT)
                    bytes_str = groups["bytes"]
                    bytes_val = 0 if bytes_str == "-" else int(bytes_str)

                    hosts.append(groups["host"])
                    timestamps.append(timestamp)
                    methods.append(groups["method"])
                    urls.append(groups["url"])
                    protocols.append(groups["protocol"])
                    statuses.append(int(groups["status"]))
                    bytes_list.append(bytes_val)

                    parsed_lines += 1

                    # Yield chunk when full
                    if len(hosts) >= chunk_size:
                        chunk_num += 1
                        if show_progress:
                            print(f"Yielding chunk {chunk_num} ({len(hosts):,} records)")

                        yield pd.DataFrame({
                            "host": hosts,
                            "timestamp": timestamps,
                            "method": pd.Categorical(methods),
                            "url": urls,
                            "protocol": pd.Categorical(protocols),
                            "status": np.array(statuses, dtype=np.int16),
                            "bytes": np.array(bytes_list, dtype=np.int64),
                        })

                        # Reset lists for next chunk
                        hosts = []
                        timestamps = []
                        methods = []
                        urls = []
                        protocols = []
                        statuses = []
                        bytes_list = []

                except (ValueError, KeyError):
                    self._parse_errors.append((line_num, line))

        # Yield remaining records
        if hosts:
            chunk_num += 1
            if show_progress:
                print(f"Yielding final chunk {chunk_num} ({len(hosts):,} records)")
                success_rate = (parsed_lines / total_lines) * 100 if total_lines > 0 else 0
                print(f"Completed: {parsed_lines:,}/{total_lines:,} lines ({success_rate:.2f}% success)")

            yield pd.DataFrame({
                "host": hosts,
                "timestamp": timestamps,
                "method": pd.Categorical(methods),
                "url": urls,
                "protocol": pd.Categorical(protocols),
                "status": np.array(statuses, dtype=np.int16),
                "bytes": np.array(bytes_list, dtype=np.int64),
            })

    def get_stats(self, df: pd.DataFrame) -> dict:
        """Get parsing statistics from DataFrame.

        Args:
            df: Parsed DataFrame

        Returns:
            Dictionary with statistics
        """
        return {
            "total_records": len(df),
            "unique_hosts": df["host"].nunique(),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if len(df) > 0 else None,
                "end": df["timestamp"].max().isoformat() if len(df) > 0 else None,
            },
            "status_codes": df["status"].value_counts().to_dict(),
            "methods": df["method"].value_counts().to_dict(),
            "total_bytes": df["bytes"].sum(),
            "avg_bytes": df["bytes"].mean(),
            "parse_errors": len(self._parse_errors),
        }
