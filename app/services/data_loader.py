"""Data loading and parsing service."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class LoadedData:
    """Container for loaded data."""
    
    data: np.ndarray
    original_length: int
    source_type: Literal["csv", "txt", "sample", "manual"]
    file_path: Path | None = None
    file_name: str | None = None
    loaded_at: datetime | None = None
    
    @property
    def length(self) -> int:
        return len(self.data)
    
    @property
    def time_range_hours(self) -> float:
        """Assuming 5-minute intervals. Uses original_length for accurate timespan."""
        return self.original_length * 5 / 60
    
    @property
    def time_range_days(self) -> float:
        """Time span in days based on original data length."""
        return self.time_range_hours / 24


class DataLoader:
    """Service for loading and validating data from various sources."""
    
    MAX_FILE_SIZE_MB = 500
    MAX_ROWS = 10_000_000
    
    def __init__(self, uploads_dir: Path):
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv(self, uploaded_file) -> LoadedData | None:
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
        
        # Extract load column
        if "load" in df.columns:
            loads = df["load"].values
        elif "request_count" in df.columns:
            loads = df["request_count"].values
        else:
            cols = ", ".join(df.columns.tolist())
            st.error(f"Need 'load' or 'request_count' column. Found: {cols}")
            return None
        
        # Validate
        loads = self._validate_data(loads)
        if loads is None:
            return None
        
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
        )
    
    def load_txt(self, uploaded_file) -> LoadedData | None:
        """Load data from TXT file."""
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
            
            loads = []
            lines = content.strip().split('\n')
            
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
            
            st.success(f"Loaded {len(loads):,} data points from TXT")
            
            return LoadedData(
                data=loads,
                original_length=len(loads),
                source_type="txt",
                file_path=saved_path,
                file_name=uploaded_file.name,
                loaded_at=datetime.now(),
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
        
        return LoadedData(
            data=loads,
            original_length=n_periods,
            source_type="sample",
            loaded_at=datetime.now(),
        )
    
    def parse_manual(self, text: str) -> LoadedData | None:
        """Parse manual comma-separated input."""
        try:
            loads = np.array([float(x.strip()) for x in text.split(",")])
            loads = self._validate_data(loads)
            if loads is None:
                return None
            
            return LoadedData(
                data=loads,
                original_length=len(loads),
                source_type="manual",
                loaded_at=datetime.now(),
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
