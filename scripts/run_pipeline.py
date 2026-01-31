"""Run the complete Phase 1 data pipeline.

This script:
1. Parses raw log files
2. Cleans the data
3. Aggregates to multiple time windows (1m, 5m, 15m)
4. Generates features
5. Saves processed files to DATA/processed/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd

from src.data.parser import LogParser
from src.data.cleaner import DataCleaner
from src.data.aggregator import TimeAggregator, AggregationConfig
from src.features.time_features import TimeFeatureExtractor
from src.features.lag_features import LagFeatureExtractor
from src.features.rolling_features import RollingFeatureExtractor
from src.features.advanced_features import AdvancedFeatureExtractor


def main():
    """Run complete data pipeline."""
    # Paths
    DATA_DIR = project_root / "DATA"
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: DATA PIPELINE")
    print("=" * 60)

    # Step 1: Parse raw logs
    print("\n[1/5] Parsing raw log files...")
    parser = LogParser()

    train_path = DATA_DIR / "train.txt"
    test_path = DATA_DIR / "test.txt"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found!")
        return

    df_train_raw = parser.parse_to_dataframe(train_path)
    print(f"  Train: {len(df_train_raw):,} records parsed")

    if test_path.exists():
        df_test_raw = parser.parse_to_dataframe(test_path)
        print(f"  Test: {len(df_test_raw):,} records parsed")
    else:
        df_test_raw = None
        print("  Test file not found, skipping...")

    # Step 2: Clean data
    print("\n[2/5] Cleaning data...")
    cleaner = DataCleaner()

    df_train_clean, train_report = cleaner.clean(df_train_raw)
    print(f"  Train: {train_report}")
    cleaner.save(df_train_clean, PROCESSED_DIR / "cleaned_train.parquet")

    if df_test_raw is not None:
        df_test_clean, test_report = cleaner.clean(df_test_raw)
        print(f"  Test: {test_report}")
        cleaner.save(df_test_clean, PROCESSED_DIR / "cleaned_test.parquet")
    else:
        df_test_clean = None

    # Step 3: Aggregate to multiple windows
    print("\n[3/5] Aggregating to time windows...")
    aggregator = TimeAggregator()

    for window in ["1min", "5min", "15min"]:
        config = AggregationConfig(window=window)
        agg = TimeAggregator(config)

        df_train_agg = agg.aggregate(df_train_clean)
        agg.save(df_train_agg, PROCESSED_DIR / f"train_{window.replace('min', 'm')}.parquet")
        print(f"  Train {window}: {len(df_train_agg):,} rows")

        if df_test_clean is not None:
            df_test_agg = agg.aggregate(df_test_clean)
            agg.save(df_test_agg, PROCESSED_DIR / f"test_{window.replace('min', 'm')}.parquet")
            print(f"  Test {window}: {len(df_test_agg):,} rows")

    # Step 4: Add features (focus on 5m window)
    print("\n[4/5] Engineering features (5-minute window)...")

    # Load 5m aggregated data
    df_train = pd.read_parquet(PROCESSED_DIR / "train_5m.parquet")
    if df_test_clean is not None:
        df_test = pd.read_parquet(PROCESSED_DIR / "test_5m.parquet")
    else:
        df_test = None

    # Time features
    time_extractor = TimeFeatureExtractor(cyclical=True)
    df_train = time_extractor.transform(df_train)
    if df_test is not None:
        df_test = time_extractor.transform(df_test)
    print(f"  Added {len(time_extractor.get_feature_names())} time features")

    # Lag features
    lag_extractor = LagFeatureExtractor(window="5min")
    df_train = lag_extractor.transform_with_diff(df_train, target_cols=["request_count"])
    df_train = lag_extractor.transform(df_train, target_cols=["bytes_total"])
    if df_test is not None:
        df_test = lag_extractor.transform_with_diff(df_test, target_cols=["request_count"])
        df_test = lag_extractor.transform(df_test, target_cols=["bytes_total"])
    print(f"  Added lag features")

    # Rolling features
    rolling_extractor = RollingFeatureExtractor(time_window="5min")
    df_train = rolling_extractor.transform(df_train, target_cols=["request_count", "bytes_total"])
    df_train = rolling_extractor.transform_ewm(df_train, target_cols=["request_count"])
    if df_test is not None:
        df_test = rolling_extractor.transform(df_test, target_cols=["request_count", "bytes_total"])
        df_test = rolling_extractor.transform_ewm(df_test, target_cols=["request_count"])
    print(f"  Added rolling features")

    # Advanced features
    advanced_extractor = AdvancedFeatureExtractor(spike_threshold=3.0, window=15)
    df_train = advanced_extractor.transform(df_train)
    if df_test is not None:
        df_test = advanced_extractor.transform(df_test)
    print(f"  Added advanced features")

    # Handle NaN values
    numeric_cols = df_train.select_dtypes(include=["number"]).columns
    df_train[numeric_cols] = df_train[numeric_cols].ffill().bfill()
    if df_test is not None:
        df_test[numeric_cols] = df_test[numeric_cols].ffill().bfill()

    # Step 5: Save feature-engineered datasets
    print("\n[5/5] Saving final datasets...")
    df_train.to_parquet(PROCESSED_DIR / "train_features_5m.parquet", index=False)
    print(f"  Saved train_features_5m.parquet: {df_train.shape}")

    if df_test is not None:
        df_test.to_parquet(PROCESSED_DIR / "test_features_5m.parquet", index=False)
        print(f"  Saved test_features_5m.parquet: {df_test.shape}")

    # Save feature info
    feature_categories = {
        "time": time_extractor.get_feature_names(),
        "lag": [c for c in df_train.columns if "lag_" in c or "diff_" in c or "pct_change" in c],
        "rolling": [c for c in df_train.columns if "rolling_" in c or "ewm_" in c],
        "advanced": [c for c in df_train.columns if any(x in c for x in
            ["spike", "trend", "velocity", "momentum", "bb_", "cv", "direction", "streak"])],
        "aggregation": ["request_count", "unique_hosts", "error_count", "error_rate",
                        "success_rate", "bytes_total", "bytes_avg", "bytes_max",
                        "requests_per_host", "bytes_per_request"],
    }

    feature_info = {
        "all_columns": list(df_train.columns),
        "numeric_features": list(df_train.select_dtypes(include=["number"]).columns),
        "target": "request_count",
        "timestamp_col": "timestamp",
        "feature_categories": {k: [f for f in v if f in df_train.columns]
                               for k, v in feature_categories.items()},
    }

    with open(PROCESSED_DIR / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    print("  Saved feature_info.json")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nProcessed files in: {PROCESSED_DIR}")
    print("\nFiles created:")
    for f in sorted(PROCESSED_DIR.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
