# Autoscaling Analysis for NASA Web Server Logs

**DATAFLOW 2026 Competition Project**

This project analyzes NASA Kennedy Space Center web server logs to predict traffic patterns and recommend autoscaling decisions.

## Project Overview

### Objectives
1. **Traffic Prediction**: Forecast request volume using time series models (SARIMA, Prophet, LightGBM)
2. **Autoscaling Optimization**: Recommend server scaling decisions to minimize cost while maintaining SLA
3. **Anomaly Detection**: Identify unusual traffic patterns using statistical and ML methods

### Dataset
- **Source**: NASA Kennedy Space Center WWW Server Logs (July-August 1995)
- **Size**: 3.46 million requests (~359 MB)
- **Training**: July 1 - August 22 (2.93M records)
- **Testing**: August 23 - August 31 (527K records)

## Key Features

### Advanced Anomaly Detection
- **Statistical Method**: Z-score based spike detection (3-sigma rule)
- **ML Method**: IsolationForest for unsupervised anomaly detection (1% contamination)
- **Dual Validation**: Both methods compared for high-confidence anomaly identification

### Domain Knowledge Integration
- **Special Events Dictionary**: 15+ historical events identified
  - US Holidays (July 4 Independence Day)
  - NASA Space Missions (STS-70 Launch, July 13-22)
  - Apollo 11 Anniversary (July 20)
  - Hurricane outage (August 1-3)
- **Event Impact Classification**: high_traffic, low_traffic, outage

### Comprehensive Feature Engineering
| Category | Count | Examples |
|----------|-------|----------|
| Time | 23 | hour, day_of_week, is_weekend, cyclical encodings |
| Lag | 24 | lag_1 to lag_288, diffs, pct_changes |
| Rolling | 16 | mean, std, min, max over multiple windows |
| Advanced | 12 | spike_score, trend, velocity, momentum |
| Aggregation | 10 | request_count, bytes_total, error_rate |

## Installation

### Requirements
- Python 3.10+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/autoscaling-analysis.git
cd autoscaling-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For MLOps features
pip install -e ".[mlops]"

# For deep learning (LSTM)
pip install -e ".[deep]"
```

## Project Structure

```
datafollow/
├── DATA/
│   ├── train.txt              # Raw training logs
│   ├── test.txt               # Raw test logs
│   └── processed/             # Processed parquet files (1m, 5m, 15m)
├── notebooks/
│   ├── 01_data_ingestion.ipynb     # Parse raw logs
│   ├── 02_aggregation.ipynb        # Time aggregation
│   ├── 03_feature_engineering.ipynb # Feature extraction + anomaly detection
│   ├── 04_eda.ipynb                # EDA with visualizations
│   ├── 05_feature_selection.ipynb  # Feature importance
│   ├── 06_baseline_models.ipynb    # SARIMA baseline
│   ├── 07_ml_models.ipynb          # Prophet + LightGBM
│   ├── 08_scaling_policy.ipynb     # Autoscaling policies
│   ├── 09_cost_simulation.ipynb    # Cost analysis
│   ├── 10_policy_optimization.ipynb # Policy tuning
│   └── 11_final_benchmark.ipynb    # Comprehensive benchmarks
├── src/
│   ├── data/                  # Data processing (parser, aggregator, cleaner)
│   ├── features/              # Feature engineering + anomaly_detector.py
│   ├── models/                # SARIMA, Prophet, LightGBM
│   ├── scaling/               # ScalingPolicy, CostSimulator
│   ├── utils/                 # Metrics, visualization
│   └── api/                   # FastAPI endpoints
├── app/
│   └── dashboard.py           # Streamlit dashboard
├── models/                    # Saved models (.pkl) + results (.json)
├── tests/                     # Unit tests (80%+ coverage)
├── reports/
│   ├── benchmark_results.csv  # Final benchmark results
│   └── figures/               # Generated visualizations
└── docs/                      # Documentation
```

## Usage

### 1. Run Data Pipeline

```bash
# Run notebooks in order
jupyter notebook notebooks/01_data_ingestion.ipynb
jupyter notebook notebooks/02_aggregation.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
```

### 2. Train Models

```bash
jupyter notebook notebooks/06_baseline_models.ipynb
jupyter notebook notebooks/07_ml_models.ipynb
```

### 3. Start API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Start Dashboard

```bash
streamlit run app/dashboard.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forecast` | POST | Predict traffic for next N minutes |
| `/recommend-scaling` | POST | Get scaling recommendation |
| `/metrics` | GET | Current model metrics |
| `/cost-report` | GET | Cost analysis |

### Example

```bash
# Get forecast
curl -X POST "http://localhost:8000/forecast?horizon=30"

# Get scaling recommendation
curl -X POST "http://localhost:8000/recommend-scaling" \
  -H "Content-Type: application/json" \
  -d '{"predicted_load": [120, 150, 180], "current_servers": 2}'
```

## Models

### Benchmark Results (5-minute granularity, Test Set)

| Model | RMSE | MAE | MAPE (%) | Use Case |
|-------|------|-----|----------|----------|
| **Prophet** | 139.19 | 102.52 | 66.74 | Best overall, handles seasonality |
| SARIMA | 150.37 | 108.56 | 58.51 | Statistical baseline, interpretable |
| LightGBM | 262.65 | 235.24 | 123.06 | Feature importance, fast inference |

**Best Model**: Prophet (lowest RMSE on test set)

### Model Descriptions
- **SARIMA**: Statistical time series model with seasonal components
- **Prophet**: Facebook's additive model for time series with holidays
- **LightGBM**: Gradient boosting with 90+ engineered features

## Development

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Docker

```bash
# Build
docker-compose build

# Run
docker-compose up

# Access
# API: http://localhost:8000
# Dashboard: http://localhost:8501
# MLflow: http://localhost:5000
```

## License

MIT License

## Team

Team Datafollow - DATAFLOW 2026 Competition
