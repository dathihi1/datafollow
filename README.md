# 🚀 NASA Traffic Autoscaling Dashboard

**DATAFLOW 2026 Competition Project**

Advanced autoscaling analysis and prediction system for NASA Kennedy Space Center web server traffic. Features a hybrid Streamlit dashboard with historical analysis and ML-powered predictive planning.

## 🌟 Features

### Dual-Mode Dashboard
- **📊 Historical Analysis Mode**: Analyze past traffic data with interactive visualizations
  - Upload CSV/TXT files (up to 500MB, auto-detects NASA log format)
  - Multiple scaling configurations (Conservative, Balanced, Aggressive)
  - Three policy types (Balanced, Reactive, Predictive) with distinct behaviors
  - Real-time cost simulation and SLA tracking
  - Export detailed reports

- **🔮 Predictive Planning Mode**: AI-powered traffic forecasting
  - Multi-model forecasting (Prophet, SARIMA, LightGBM, Ensemble)
  - 7-30 day forecast horizons with confidence intervals
  - **Iterative forecasting** with trend analysis and realistic variations
  - **Real timestamp continuation** from historical data (not system time)
  - Automated configuration recommendations
  - **9-scenario comparison matrix** (3 configs × 3 policies)
  - What-if scenario analysis with traffic multipliers and spike simulation
  - Cost optimization with risk assessment

### Smart Features
- **NASA Log Auto-detection**: Automatically parses Apache Combined Log Format
- **Intelligent Time Interval Calculation**: Extracts actual intervals from log timestamps
- **Intelligent Downsampling**: LTTB algorithm for smooth visualization of large datasets (millions of points)
- **Persistent Data**: Uploaded files cached across mode changes with timestamp tracking
- **Real-time Metrics**: Live cost calculations based on AWS EC2 pricing ($0.85/server/hour)
- **Anomaly Detection**: Statistical and ML-based traffic spike identification

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

## 📦 Installation

### Requirements
- Python 3.10+
- Virtual environment (venv/conda)

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-repo/datafollow.git
cd datafollow

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The dashboard uses `.streamlit/config.toml` for server settings:
- **Max Upload Size**: 500MB
- **Auto-reload**: Enabled for development
- **Port**: 8502 (configurable)

## 📁 Project Structure

```
datafollow/
├── app/
│   ├── dashboard_v2.py           # 🆕 Main hybrid dashboard (run this!)
│   ├── components/               # 🆕 Modular UI components
│   │   ├── sidebar.py           # Configuration sidebar
│   │   ├── charts.py            # Shared visualizations
│   │   ├── historical.py        # Historical analysis tabs
│   │   └── predictive.py        # Predictive planning tabs
│   └── services/                # 🆕 Business logic services
│       ├── data_loader.py       # Data loading & validation
│       ├── model_service.py     # ML model management
│       ├── simulator_service.py # Scaling simulation
│       └── recommendation_service.py # AI recommendations
├── DATA/
│   ├── train.txt                # Raw training logs
│   ├── test.txt                 # Raw test logs
│   ├── uploads/                 # 🆕 User uploaded files
│   └── processed/               # Processed parquet files
├── models/
│   ├── lgbm_5m.pkl             # LightGBM model (91 features)
│   ├── prophet_5m.pkl          # Prophet model
│   ├── sarima_5m.pkl           # SARIMA model
│   ├── feature_scaler.pkl      # RobustScaler for features
│   └── *.json                  # Model configs & results
├── src/
│   ├── data/                   # Data processing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model implementations
│   ├── scaling/                # 🆕 ScalingConfig, Policy, Simulator
│   ├── utils/                  # Utilities
│   └── api/                    # FastAPI endpoints
├── notebooks/                  # Jupyter analysis notebooks (01-11)
├── tests/                      # Unit tests
├── .streamlit/
│   └── config.toml            # 🆕 Streamlit configuration
└── requirements.txt           # Python dependencies
```

**🆕 = New/Updated in latest version**

## 🚀 Usage

### Start the Dashboard

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the hybrid dashboard
streamlit run app/dashboard_v2.py

# Custom port (optional)
streamlit run app/dashboard_v2.py --server.port 8502
```

Access at: **http://localhost:8501** (or your custom port)

### Dashboard Workflow

#### Historical Analysis Mode
1. **Load Data**: Upload CSV/TXT or use sample data
2. **Configure**: Select preset (Conservative/Balanced/Aggressive) and policy
3. **Analyze**: View traffic patterns, scaling behavior, cost metrics
4. **Export**: Download detailed reports

#### Predictive Planning Mode
1. **Load Historical Data**: Upload past traffic data (CSV/TXT/NASA logs)
2. **Select Model**: Choose LightGBM/Prophet/SARIMA/Ensemble
3. **Generate Forecast**: Set horizon (7-30 days) and confidence level
   - Forecasts continue from last data timestamp (not current system time)
   - Iterative forecasting with trend and realistic daily variations
4. **Run Simulation**: Click "Run All Simulations" to test 9 scenarios (3 configs × 3 policies)
   - View comparison matrix heatmap
   - See Best Cost, Best SLA, Best Balance winners
5. **Get Recommendations**: AI suggests optimal config based on cost/SLA priorities
6. **What-If Analysis**: Test custom scenarios with traffic multipliers and spike injection

### Data Format

**CSV Format:**
```csv
load,timestamp
1000,2023-01-01 00:00:00
1200,2023-01-01 00:05:00
...
```
- Required column: `load` or `request_count`
- Optional: `timestamp` (auto-generated if missing)
- Assumes 5-minute intervals

**TXT Format:**
```
1000
1200
1500
...
```
- One number per line, or comma-separated
- Represents request counts per 5-minute period
- **Auto-detects NASA Apache logs**: Pattern `- - [timestamp] "request"`
- Automatically aggregates to 5-minute windows from parsed timestamps

## 🔧 API Endpoints (Optional)

Start the FastAPI server for programmatic access:

```bash
uvicorn src.api.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forecast` | POST | Predict traffic for next N periods |
| `/recommend-scaling` | POST | Get scaling recommendation |
| `/metrics` | GET | Current model metrics |
| `/cost-report` | GET | Cost analysis report |

### API Example

```bash
# Get forecast
curl -X POST "http://localhost:8000/forecast?horizon=30"

# Get scaling recommendation
curl -X POST "http://localhost:8000/recommend-scaling" \
  -H "Content-Type: application/json" \
  -d '{"predicted_load": [120, 150, 180], "current_servers": 2}'
```

## 🤖 Models & Configuration

### Pre-trained Models

| Model | File | Features | Test RMSE | Best For |
|-------|------|----------|-----------|----------|
| **🏆 LightGBM** | `lgbm_5m.pkl` | 91 engineered features | **3.59** | Feature-rich data (best accuracy) |
| **Prophet** | `prophet_5m.pkl` | Seasonal patterns | 83.26 | Raw request_count, handles holidays |
| **SARIMA** | `sarima_5m.pkl` | Statistical AR/MA | 150.37 | Short-term predictions, interpretable |
| **Ensemble** | Auto-combines above | Multiple models | ~40 | Most robust predictions |

**Important**: 
- **LightGBM** achieves RMSE 3.59 when trained with 91 features (time, lag, rolling, anomaly, etc.)
- Dashboard uses simplified feature set → LightGBM falls back to seasonal forecast for raw data
- For raw `request_count` data: Use **Prophet** (RMSE 83.26) or **Ensemble**

### Scaling Configurations

| Preset | Scale Out | Scale In | Cooldown | Best For |
|--------|-----------|----------|----------|----------|
| **Conservative** | 70% @ 5 periods | 20% @ 10 periods | 10 min | Cost-sensitive, stable traffic |
| **Balanced** | 80% @ 3 periods | 30% @ 6 periods | 5 min | General use, moderate cost/SLA |
| **Aggressive** | 85% @ 2 periods | 40% @ 4 periods | 3 min | SLA-critical, high variability |

### Scaling Policies

- **Balanced**: Standard threshold-based scaling with 3-period consecutive check
  - Scale out: 3 periods @ 80% utilization
  - Moderate response time, balanced cost/SLA
- **Reactive**: Immediate response to load changes (1-period consecutive)
  - Scale out: 1 period @ 80% utilization (3x faster than Balanced)
  - Shorter cooldown (3 min vs 5 min)
  - More scaling events, higher cost, lower SLA violations
- **Predictive**: Proactive scaling using trend analysis
  - Pre-scales based on 5% upward trend detection
  - Scale out at 75% (earlier than other policies)
  - Safety margin: 15% over-provisioning
  - Highest cost, lowest SLA violations (ideal for production)

**Key Differences in Simulation Matrix:**
- **Reactive** generates more scaling events → higher cost but better SLA
- **Predictive** pre-scales before spikes → highest cost, best SLA protection  
- **Balanced** offers middle ground → most cost-effective with acceptable SLA

### Cost Model

- **Base Price**: $0.85/server/hour (AWS t3.medium equivalent)
- **Calculation**: `cost = num_servers × $0.85 × hours`
- **Example**: 10 servers × 8 days = $1,632 (full capacity) or ~$800-1,200 (with autoscaling)

## 🧪 Development

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/ app/

# Lint
ruff check src/ tests/ app/

# Type checking
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Docker Deployment

```bash
# Build containers
docker-compose build

# Start all services
docker-compose up

# Access
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
# - MLflow (optional): http://localhost:5000
```

## 📊 Performance Metrics

### Benchmark Results (5-minute granularity, NASA test set)

| Model | RMSE | MAE | MAPE (%) | R² | Speed |
|-------|------|-----|----------|-----|-------|
| **🏆 LightGBM** | **3.59** | **2.26** | **1.57%** | **0.999** | ⚡⚡⚡ Fastest |
| Prophet | 83.26 | 62.90 | 47.19% | 0.539 | ⚡ Fast |
| SARIMA | 150.37 | 108.56 | 58.51% | -0.504 | ⚡⚡ Very Fast |
| Ensemble | ~40 | ~30 | ~20% | ~0.95 | ⚡ Moderate |

**Recommendation**: Use **LightGBM** for best accuracy (RMSE 3.59 with 91 features), **Prophet** for raw request_count data (RMSE 83.26).

### Scaling Simulation Results

- **Cost Savings**: 30-45% vs. static provisioning
- **SLA Compliance**: 98%+ uptime with Balanced config
- **Response Time**: < 5 minutes average scaling latency
- **Data Handling**: Supports up to 500MB files (10M+ data points)

## 🏆 Key Technologies

- **Frontend**: Streamlit 1.31+, Plotly 5.18+
- **ML/Forecasting**: Prophet 1.1.5+, Statsmodels 0.14+, LightGBM 4.3+
- **Data Processing**: Pandas 2.2+, NumPy 1.26+, PyArrow 15+
- **Backend**: FastAPI, Uvicorn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Testing**: Pytest, pytest-cov
- **Deployment**: Docker, Docker Compose

## 📝 License

MIT License

## 👥 Contributors

**Team Datafollow** - DATAFLOW 2026 Competition

---

**Need Help?** Check the [Issues](https://github.com/your-repo/datafollow/issues) page or open a new issue.
