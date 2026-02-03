"""Data loading and parsing service."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st

# Import NASA log parser
try:
    from src.data.parser import LogParser
except ImportError:
    LogParser = None


@dataclass
class LoadedData:
    """Container for loaded data."""

    data: np.ndarray
    original_length: int
    source_type: Literal["csv", "txt", "sample", "manual"]
    file_path: Path | None = None
    file_name: str | None = None
    loaded_at: datetime | None = None
    time_interval_minutes: float | None = None  # None = unknown interval
    timestamps: list[datetime] | None = None  # Actual timestamps if available

    @property
    def length(self) -> int:
        return len(self.data)

    @property
    def time_range_hours(self) -> float | None:
        """Calculate time range if interval is known."""
        if self.time_interval_minutes is None:
            return None
        return self.original_length * self.time_interval_minutes / 60

    @property
    def time_range_days(self) -> float | None:
        """Time span in days if interval is known."""
        hours = self.time_range_hours
        if hours is None:
            return None
        return hours / 24
    
    @property
    def last_timestamp(self) -> datetime | None:
        """Get the last timestamp from the data."""
        if self.timestamps and len(self.timestamps) > 0:
            return self.timestamps[-1]
        return None


class DataLoader:
    """Service for loading and validating data from various sources."""
    
    MAX_FILE_SIZE_MB = 500
    MAX_ROWS = 10_000_000
    
    def __init__(self, uploads_dir: Path):
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def get_csv_columns(self, uploaded_file) -> tuple[list[str], list[str]]:
        """Get column names from CSV for user selection.

        Returns:
            Tuple of (numeric_columns, all_columns)
        """
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            return numeric_cols, all_cols
        except Exception:
            return [], []

    def _parse_timestamp_column(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
    ) -> tuple[float | None, float | None]:
        """Parse timestamp column and calculate time span and interval.

        Returns:
            Tuple of (total_minutes, interval_minutes) or (None, None) if failed
        """
        try:
            # Try to parse timestamps
            timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")

            # Drop NaT values
            valid_timestamps = timestamps.dropna()
            if len(valid_timestamps) < 2:
                return None, None

            # Calculate time span
            time_span = valid_timestamps.max() - valid_timestamps.min()
            total_minutes = time_span.total_seconds() / 60

            # Calculate average interval
            n_rows = len(valid_timestamps)
            interval_minutes = total_minutes / max(n_rows - 1, 1)

            return total_minutes, interval_minutes
        except Exception:
            return None, None

    def _generate_synthetic_timestamps(
        self,
        n_points: int,
        interval_minutes: float,
        reference_date: datetime | None = None,
    ) -> list[datetime]:
        """Generate synthetic timestamps for data without real timestamps.
        
        Args:
            n_points: Number of data points
            interval_minutes: Time interval between points
            reference_date: Starting date (default: use NASA data start date)
            
        Returns:
            List of datetime objects
        """
        if reference_date is None:
            # Default to NASA dataset start date (July 1, 1995)
            reference_date = datetime(1995, 7, 1, 0, 0, 0)
        
        from datetime import timedelta
        return [
            reference_date + timedelta(minutes=interval_minutes * i)
            for i in range(n_points)
        ]

    def load_csv(
        self,
        uploaded_file,
        column_name: str | None = None,
        timestamp_column: str | None = None,
        time_interval_minutes: float | None = None,
    ) -> LoadedData | None:
        """Load data from CSV file."""
        # Check file size
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                st.error(f"File too large ({file_size_mb:.1f}MB). Max: {self.MAX_FILE_SIZE_MB}MB")
                return None
            
            if file_size_mb > 50:
                st.info(f"Loading large file ({file_size_mb:.1f}MB)...")
        except AttributeError:
            pass
        
        try:
            uploaded_file.seek(0)
            
            # Chunked reading for large files
            if hasattr(uploaded_file, 'size') and uploaded_file.size > 50 * 1024 * 1024:
                chunks = []
                with st.spinner("Reading large CSV in chunks..."):
                    for chunk in pd.read_csv(uploaded_file, chunksize=100000):
                        chunks.append(chunk)
                        if sum(len(c) for c in chunks) > self.MAX_ROWS:
                            st.error(f"Too many rows. Max: {self.MAX_ROWS:,}")
                            return None
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(uploaded_file)
                
        except pd.errors.EmptyDataError:
            st.error("CSV file is empty.")
            return None
        except pd.errors.ParserError as e:
            st.error(f"CSV parse error: {e}")
            return None
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None
        
        if len(df) > self.MAX_ROWS:
            st.error(f"Too many rows ({len(df):,}). Max: {self.MAX_ROWS:,}")
            return None
        
        if len(df) == 0:
            st.error("CSV contains no data.")
            return None
        
        # Extract load column - user-selected or fallback
        if column_name and column_name in df.columns:
            loads = df[column_name].values
        elif "load" in df.columns:
            loads = df["load"].values
        elif "request_count" in df.columns:
            loads = df["request_count"].values
        else:
            cols = ", ".join(df.columns.tolist())
            st.error(f"No valid column. Select a numeric column. Found: {cols}")
            return None
        
        # Validate
        loads = self._validate_data(loads)
        if loads is None:
            return None

        # Calculate time interval from timestamp column if provided
        calculated_interval = time_interval_minutes
        total_time_minutes = None
        timestamps_list = None

        if timestamp_column and timestamp_column in df.columns:
            total_minutes, interval = self._parse_timestamp_column(df, timestamp_column)
            if total_minutes is not None and interval is not None:
                total_time_minutes = total_minutes
                calculated_interval = interval
                days = total_minutes / 60 / 24
                st.info(f"Time span: {days:.2f} days, Avg interval: {interval:.4f} min/row")
                # Try to extract actual timestamps if column is parseable
                try:
                    timestamps_list = pd.to_datetime(df[timestamp_column]).tolist()
                except Exception:
                    pass
        
        # Generate synthetic timestamps if no real timestamps available
        if timestamps_list is None and calculated_interval:
            timestamps_list = self._generate_synthetic_timestamps(
                n_points=len(loads),
                interval_minutes=calculated_interval,
            )

        # Save file
        saved_path = self._save_file(uploaded_file, "csv")

        st.success(f"Loaded {len(loads):,} data points from CSV")

        return LoadedData(
            data=loads,
            original_length=len(loads),
            source_type="csv",
            file_path=saved_path,
            file_name=uploaded_file.name,
            loaded_at=datetime.now(),
            time_interval_minutes=calculated_interval,
            timestamps=timestamps_list,
        )
    
    def load_txt(
        self,
        uploaded_file,
        time_interval_minutes: float | None = None,
        aggregation_interval_minutes: int = 5,
    ) -> LoadedData | None:
        """Load data from TXT file.

        Automatically detects NASA log format (with timestamps) and parses accordingly.
        Falls back to simple numeric parsing if not in log format.

        Args:
            uploaded_file: Streamlit uploaded file
            time_interval_minutes: Time interval for numeric data (auto-detected for NASA logs)
            aggregation_interval_minutes: Aggregation window for NASA logs (default 5min optimal)
        """
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                st.error(f"File too large ({file_size_mb:.1f}MB). Max: {self.MAX_FILE_SIZE_MB}MB")
                return None
        except AttributeError:
            pass
        
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            # Try parsing as NASA log format first
            if LogParser is not None and self._is_nasa_log_format(lines):
                st.info("Detected NASA log format - parsing timestamps...")
                return self._parse_nasa_log_format(
                    uploaded_file, content, lines, aggregation_interval_minutes
                )
            
            # Fallback: simple numeric parsing
            st.info("Parsing as numeric data...")
            loads = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    loads.append(float(line))
                except ValueError:
                    parts = line.replace(',', ' ').replace('\t', ' ').split()
                    for part in parts:
                        try:
                            loads.append(float(part))
                        except ValueError:
                            continue
            
            if not loads:
                st.error("No numeric values found in TXT file.")
                return None
            
            loads = np.array(loads)
            
            if len(loads) > self.MAX_ROWS:
                st.error(f"Too many values ({len(loads):,}). Max: {self.MAX_ROWS:,}")
                return None
            
            loads = self._validate_data(loads)
            if loads is None:
                return None
            
            saved_path = self._save_file(uploaded_file, "txt")
            
            # Ask user for time interval if not provided
            if time_interval_minutes is None:
                st.warning(
                    "Could not auto-detect time interval. "
                    "Assuming 5-minute intervals (NASA dataset default). "
                    "If incorrect, please use CSV upload with timestamp column."
                )
                time_interval_minutes = 5.0
            
            # Generate synthetic timestamps
            timestamps_list = self._generate_synthetic_timestamps(
                n_points=len(loads),
                interval_minutes=time_interval_minutes,
            )
            
            st.success(f"Loaded {len(loads):,} data points from TXT ({time_interval_minutes:.0f}-min intervals)")
            
            return LoadedData(
                data=loads,
                original_length=len(loads),
                source_type="txt",
                file_path=saved_path,
                file_name=uploaded_file.name,
                loaded_at=datetime.now(),
                time_interval_minutes=time_interval_minutes,
                timestamps=timestamps_list,
            )
            
        except UnicodeDecodeError:
            st.error("Cannot decode TXT file. Use UTF-8 encoding.")
            return None
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return None
    
    def generate_sample(self, n_periods: int = 288, seed: int = 42) -> LoadedData:
        """Generate sample load data."""
        rng = np.random.default_rng(seed)
        
        hours = np.arange(n_periods) % 288
        hour_of_day = hours / 12
        
        daily_pattern = 50 + 80 * np.sin(np.pi * (hour_of_day - 6) / 12)
        daily_pattern = np.maximum(daily_pattern, 20)
        
        spikes = np.zeros(n_periods)
        spike_indices = rng.choice(n_periods, size=10, replace=False)
        spikes[spike_indices] = rng.uniform(50, 150, size=10)
        
        noise = rng.normal(0, 10, n_periods)
        
        loads = daily_pattern + spikes + noise
        loads = np.maximum(loads, 10)
        
        # Generate synthetic timestamps
        timestamps_list = self._generate_synthetic_timestamps(
            n_points=n_periods,
            interval_minutes=5.0,
        )
        
        return LoadedData(
            data=loads,
            original_length=n_periods,
            source_type="sample",
            loaded_at=datetime.now(),
            time_interval_minutes=5.0,  # NASA-style 5-minute intervals
            timestamps=timestamps_list,
        )

    def parse_manual(self, text: str) -> LoadedData | None:
        """Parse manual comma-separated input."""
        try:
            loads = np.array([float(x.strip()) for x in text.split(",")])
            loads = self._validate_data(loads)
            if loads is None:
                return None
            
            # Generate synthetic timestamps (assume 5-min intervals)
            timestamps_list = self._generate_synthetic_timestamps(
                n_points=len(loads),
                interval_minutes=5.0,
            )
            
            return LoadedData(
                data=loads,
                original_length=len(loads),
                source_type="manual",
                loaded_at=datetime.now(),
                time_interval_minutes=5.0,
                timestamps=timestamps_list,
            )
        except ValueError:
            st.error("Invalid format. Use comma-separated numbers.")
            return None
    
    def list_saved_files(self) -> list[Path]:
        """List all saved upload files."""
        files = []
        for ext in ["*.csv", "*.txt"]:
            files.extend(self.uploads_dir.glob(ext))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    
    def load_saved_file(self, file_path: Path) -> LoadedData | None:
        """Load a previously saved file."""
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            if "load" in df.columns:
                loads = df["load"].values
            elif "request_count" in df.columns:
                loads = df["request_count"].values
            else:
                st.error("Saved file missing required column")
                return None
        else:
            with open(file_path, 'r') as f:
                content = f.read()
            loads = []
            for line in content.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    loads.append(float(line))
                except ValueError:
                    parts = line.replace(',', ' ').split()
                    for part in parts:
                        try:
                            loads.append(float(part))
                        except ValueError:
                            continue
            loads = np.array(loads)
        
        loads = self._validate_data(loads)
        if loads is None:
            return None
        
        return LoadedData(
            data=loads,
            original_length=len(loads),
            source_type="csv" if file_path.suffix == ".csv" else "txt",
            file_path=file_path,
            file_name=file_path.name,
            loaded_at=datetime.now(),
        )
    
    def _validate_data(self, loads: np.ndarray) -> np.ndarray | None:
        """Validate and clean loaded data."""
        if np.any(np.isnan(loads)):
            st.warning("NaN values replaced with 0")
            loads = np.nan_to_num(loads, nan=0.0)
        
        if np.any(loads < 0):
            st.warning("Negative values set to 0")
            loads = np.maximum(loads, 0)
        
        return loads
    
    def _save_file(self, uploaded_file, file_type: str) -> Path | None:
        """Save uploaded file to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{uploaded_file.name}"
            filepath = self.uploads_dir / filename
            
            uploaded_file.seek(0)
            with open(filepath, 'wb') as f:
                f.write(uploaded_file.read())
            
            uploaded_file.seek(0)
            return filepath
        except Exception as e:
            st.warning(f"Could not save file: {e}")
            return None

    def _is_nasa_log_format(self, lines: list[str]) -> bool:
        """Check if file appears to be in NASA log format.
        
        Args:
            lines: First few lines of the file
            
        Returns:
            True if looks like NASA log format
        """
        if not lines:
            return False
        
        # Check first non-empty line
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            # NASA logs have pattern: host - - [timestamp] "request" status bytes
            if ' - - [' in line and '] "' in line:
                return True
        return False

    def _parse_nasa_log_format(
        self,
        uploaded_file,
        content: str,
        lines: list[str],
        aggregation_interval_minutes: int = 5,
    ) -> LoadedData | None:
        """Parse NASA log format and aggregate by time window.

        Args:
            uploaded_file: Streamlit file object
            content: File content string
            lines: Split lines
            aggregation_interval_minutes: Aggregation window (default 5min optimal for ML)

        Returns:
            LoadedData with aggregated request counts
        """
        try:
            parser = LogParser()
            
            # Parse all entries
            entries = []
            for line in lines:
                if not line.strip():
                    continue
                entry = parser.parse_line(line)
                if entry is not None:
                    entries.append(entry)
            
            if not entries:
                st.error("Could not parse any NASA log entries")
                return None
            
            # Convert to dataframe with timestamps
            df = pd.DataFrame([{
                'timestamp': e.timestamp,
                'bytes': e.bytes,
            } for e in entries])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate actual time span and interval
            total_seconds = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            total_minutes = total_seconds / 60
            n_entries = len(df)
            avg_interval_minutes = total_minutes / max(n_entries - 1, 1)
            
            # Aggregate to user-selected time window (default 5min is optimal for ML)
            freq = f'{aggregation_interval_minutes}min'
            df['time_bucket'] = df['timestamp'].dt.floor(freq)
            aggregated = df.groupby('time_bucket').size().reset_index(name='request_count')

            # Fill missing time buckets with 0
            full_time_range = pd.date_range(
                start=aggregated['time_bucket'].min(),
                end=aggregated['time_bucket'].max(),
                freq=freq,
            )
            aggregated = aggregated.set_index('time_bucket').reindex(
                full_time_range, fill_value=0
            ).reset_index()
            aggregated.columns = ['timestamp', 'request_count']

            loads = aggregated['request_count'].values
            timestamps = aggregated['timestamp'].tolist()
            time_interval = float(aggregation_interval_minutes)

            # Safety check: downsample if still too many rows
            if len(loads) > self.MAX_ROWS:
                st.warning(
                    f"Aggregated data has {len(loads):,} rows. "
                    f"Downsampling to 15-minute intervals..."
                )
                df['time_bucket'] = df['timestamp'].dt.floor('15min')
                aggregated = df.groupby('time_bucket').size().reset_index(name='request_count')
                full_time_range = pd.date_range(
                    start=aggregated['time_bucket'].min(),
                    end=aggregated['time_bucket'].max(),
                    freq='15min',
                )
                aggregated = aggregated.set_index('time_bucket').reindex(
                    full_time_range, fill_value=0
                ).reset_index()
                aggregated.columns = ['timestamp', 'request_count']
                loads = aggregated['request_count'].values
                timestamps = aggregated['timestamp'].tolist()
                time_interval = 15.0
            
            loads = self._validate_data(loads)
            if loads is None:
                return None
            
            saved_path = self._save_file(uploaded_file, "txt")
            
            days = total_minutes / 60 / 24
            st.success(
                f"Parsed {n_entries:,} NASA log entries\n\n"
                f"Time span: {days:.2f} days ({df['timestamp'].min()} to {df['timestamp'].max()})\n\n"
                f"Aggregated to {len(loads):,} time buckets ({time_interval:.0f}-minute intervals)"
            )
            
            return LoadedData(
                data=loads,
                original_length=len(loads),
                source_type="txt",
                file_path=saved_path,
                file_name=uploaded_file.name,
                loaded_at=datetime.now(),
                time_interval_minutes=time_interval,
                timestamps=timestamps,
            )
            
        except Exception as e:
            st.error(f"Error parsing NASA log format: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
