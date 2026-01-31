# 🚀 AUTOSCALING ANALYSIS - PROJECT PLAN
**Cuộc thi: DATAFLOW 2026**  
**Chủ đề: Autoscaling Analysis cho NASA Web Server Logs**

---

## 📋 MỤC LỤC
1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Phân tích dữ liệu](#2-phân-tích-dữ-liệu)
3. [Feature Engineering](#3-feature-engineering)
4. [Kế hoạch thực hiện](#4-kế-hoạch-thực-hiện)
5. [Kiến trúc hệ thống](#5-kiến-trúc-hệ-thống)
6. [Timeline & Checklist](#6-timeline--checklist)

---

## 1. TỔNG QUAN BÀI TOÁN

### 1.1 Bối cảnh
Trong quản trị hệ thống đám mây, việc cấp phát tài nguyên cố định dẫn đến:
- ❌ **Lãng phí tài nguyên** khi ít người truy cập
- ❌ **Sập hệ thống** khi lượng truy cập tăng đột biến

### 1.2 Mục tiêu
Xây dựng hệ thống phân tích nhật ký truy cập để:

| Mục tiêu | Mô tả |
|----------|-------|
| **Bài toán Hồi quy** | Dự báo lưu lượng truy cập (số request, bytes) trong tương lai |
| **Bài toán Tối ưu** | Tự động điều chỉnh số lượng server (Autoscaling) để tối ưu chi phí |

### 1.3 Dữ liệu
- **Nguồn**: NASA Kennedy Space Center WWW Server Logs
- **Thời gian**: Tháng 7-8/1995 (62 ngày)
- **Quy mô**: 3.46 triệu requests (~359 MB)

#### Train/Test Split
| Tập dữ liệu | Khoảng thời gian | Số lượng |
|-------------|------------------|----------|
| **Train** | Jul 1 - Aug 22 (53 ngày) | 2,934,961 records |
| **Test** | Aug 23 - Aug 31 (9 ngày) | 526,651 records |

#### Lưu ý đặc biệt
- ⚠️ **Missing data**: Aug 1 14:52 - Aug 3 04:36 (do bão)
- ✅ **Parse rate**: 100% (không có log bị lỗi format)

### 1.4 Deliverables
| Loại | Yêu cầu |
|------|---------|
| **Mô hình ML** | Tối thiểu 2 mô hình (ARIMA/Prophet/LSTM/XGBoost) |
| **Metrics** | RMSE, MSE, MAE, MAPE |
| **Khung thời gian** | 1m, 5m, 15m aggregation |
| **Scaling Policy** | Logic rules + cost analysis |
| **Demo** | API (FastAPI) + Dashboard (Streamlit) |
| **Documentation** | Báo cáo (max 30 trang), README, slides |
| **Video** | 3-5 phút demo |

---

## 2. PHÂN TÍCH DỮ LIỆU

### 2.1 Thống kê tổng quan

```
📊 DATASET OVERVIEW
├── Train records: 2,934,961
├── Test records: 526,651
├── Total records: 3,461,612
├── Unique hosts: ~19,000+
├── Date range: Jul 1, 1995 - Aug 31, 1995
└── Parse success rate: 100%
```

### 2.2 Cấu trúc Log (Apache Combined Format)

```
Format: <host> - - [<timestamp>] "<request>" <status> <bytes>
Example: 199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
```

#### Raw Fields
| Field | Ví dụ | Type | Mô tả |
|-------|-------|------|-------|
| `host` | `199.72.81.55` | String | IP/domain của client |
| `timestamp` | `01/Jul/1995:00:00:01 -0400` | Datetime | Thời điểm request |
| `method` | `GET` | Categorical | HTTP method |
| `url` | `/history/apollo/` | String | Đường dẫn resource |
| `protocol` | `HTTP/1.0` | String | Giao thức |
| `status` | `200` | Integer | Mã phản hồi HTTP |
| `bytes` | `6245` | Integer | Dung lượng response |

### 2.3 Phân bố thời gian

#### 📅 Hourly Pattern (EST timezone)
```
00:00 | ################### 950      (Low traffic)
04:00 | ######### 497                (Lowest - 4AM)
08:00 | ######################### 1257
12:00 | ###################################### 1936  (Peak - Noon)
16:00 | ##################################### 1854   (Peak continues)
20:00 | ##################### 1094
23:00 | #################### 1047
```

**Insights:**
- 🔴 Peak hours: 11:00-17:00 EST (giờ làm việc tại Mỹ)
- 🔵 Low hours: 03:00-06:00 EST
- 📈 Pattern: Business hours spike (rõ ràng cho weekly seasonality)

#### 📆 Daily Trends
```
Jul 1-7   : Stable (~600-900 req/sample)
Jul 8-9   : Drop (~350) - Weekend
Jul 13    : SPIKE (1342) - Sự kiện đặc biệt?
Jul 15-16 : Low (~450) - Weekend
Jul 22-23 : Low (~350) - Weekend
Aug 1-3   : Missing data (Hurricane)
```

**Insights:**
- 📊 Weekly seasonality rõ ràng
- 🏖️ Weekend drop ~50%
- ⚡ Spike detection: Jul 13 (cần investigate)
- ⚠️ Missing data handling: Imputation vs. exclusion

#### Top Traffic Spikes
| Timestamp | Estimated Requests/min |
|-----------|------------------------|
| 1995-07-13 09:10 | ~400 |
| 1995-07-13 09:13 | ~400 |
| 1995-07-13 09:49 | ~400 |
| 1995-07-13 08:25-08:46 | ~300 (sustained) |

### 2.4 HTTP Status Codes

| Code | Count | % | Meaning |
|------|-------|---|---------|
| 200 | 9013 | 90.13% | Success |
| 304 | 531 | 5.31% | Not Modified (cache hit) |
| 302 | 408 | 4.08% | Redirect |
| 404 | 48 | 0.48% | Not Found |

**Insights:**
- ✅ Error rate rất thấp (< 0.5%)
- 💾 Cache efficiency tốt (5.3% 304 responses)

### 2.5 Content Analysis

#### HTTP Methods
```
GET:  99.85%
HEAD: 0.14%
POST: 0.01%
```

#### URL Categories
| Category | % | Examples |
|----------|---|----------|
| Images | 33.6% | `/images/*.gif` |
| Shuttle | 33.0% | `/shuttle/missions/*` |
| History | 15.0% | `/history/apollo/*` |
| Other | 13.5% | Root, misc |
| Software | 1.8% | `/software/winvn/*` |
| CGI-bin | 1.7% | `/cgi-bin/imagemap/*` |

#### Content Types (by extension)
| Type | % | Impact |
|------|---|--------|
| GIF images | 56.5% | High bandwidth |
| HTML | 22.4% | Medium bandwidth |
| JPEG | 2.6% | High bandwidth |
| Videos (MPG) | 1.2% | Very high bandwidth |

### 2.6 Response Size Statistics

```
📦 SIZE DISTRIBUTION
├── Mean:     18 KB
├── Median:   3.6 KB
├── Std Dev:  64 KB
├── Max:      1.2 MB
└── Min:      0 bytes (7.6% empty responses)

BUCKETS:
├── 0 bytes (empty):  7.6%
├── < 1 KB:          22.5%
├── 1-10 KB:         44.4%  ← Most common
├── 10-100 KB:       22.5%
├── 100KB-1MB:        3.0%
└── > 1 MB:           0.1%
```

### 2.7 Bytes Traffic Pattern by Hour

```
00:00 | # 18.0 MB
08:00 | ## 20.7 MB
12:00 | ### 34.7 MB  (Peak)
16:00 | ### 33.7 MB
17:00 | ### 35.1 MB  (Peak)
23:00 | ## 21.1 MB
```

**Insights:**
- 💾 Peak bandwidth: 12:00-17:00
- 📊 Bandwidth pattern tương đồng request pattern
- ⚡ Scaling cần xem xét cả request count và bytes

---

## 3. FEATURE ENGINEERING

### 3.1 Level 1: Time Features (Bắt buộc)

| Feature | Formula | Type | Purpose |
|---------|---------|------|---------|
| `timestamp` | Original | Datetime | Index |
| `year` | Extract from timestamp | Integer | Long-term trend |
| `month` | Extract from timestamp | Integer | Monthly seasonality |
| `day` | Extract from timestamp | Integer | Daily trend |
| `hour` | Extract from timestamp | Integer [0-23] | Hourly pattern |
| `minute` | Extract from timestamp | Integer [0-59] | Intra-hour pattern |
| `day_of_week` | Monday=0, Sunday=6 | Integer [0-6] | Weekly seasonality |
| `is_weekend` | 1 if Sat/Sun else 0 | Binary | Weekend effect |
| `is_business_hour` | 1 if 8-18h else 0 | Binary | Office hours |
| `time_of_day` | morning/afternoon/evening/night | Categorical | Day segment |

**Code example:**
```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_business_hour'] = df['hour'].between(8, 18).astype(int)
```

### 3.2 Level 2: Aggregation Features (Core)

Aggregate log entries theo khung thời gian:

| Aggregation | Window | Metrics |
|-------------|--------|---------|
| **1 minute** | 1m | `request_count_1m`, `bytes_sum_1m`, `unique_hosts_1m` |
| **5 minutes** | 5m | `request_count_5m`, `bytes_sum_5m`, `error_rate_5m` |
| **15 minutes** | 15m | `request_count_15m`, `bytes_sum_15m`, `avg_bytes_15m` |

**Aggregation metrics:**
```python
agg_funcs = {
    'host': 'nunique',           # unique_hosts
    'bytes': ['sum', 'mean'],    # total & average bandwidth
    'status': [
        ('error_count', lambda x: (x >= 400).sum()),
        ('success_rate', lambda x: (x == 200).mean())
    ]
}
```

### 3.3 Level 3: Lag Features (Time Series)

| Feature | Description | Formula |
|---------|-------------|---------|
| `requests_lag_1` | Previous period | `shift(1)` |
| `requests_lag_5` | 5 periods ago | `shift(5)` |
| `requests_lag_12` | 12 periods ago (1h for 5m) | `shift(12)` |
| `requests_lag_60` | 1 hour ago (for 1m data) | `shift(60)` |
| `requests_lag_288` | Same time yesterday (for 5m) | `shift(288)` |
| `requests_lag_2016` | Same time last week (for 5m) | `shift(2016)` |

**Code example:**
```python
for lag in [1, 5, 12, 60, 288, 2016]:
    df[f'requests_lag_{lag}'] = df['request_count'].shift(lag)
```

### 3.4 Level 4: Rolling Statistics

| Feature | Window | Description |
|---------|--------|-------------|
| `requests_rolling_mean_5` | 5 periods | Short-term average |
| `requests_rolling_mean_15` | 15 periods | Medium-term average |
| `requests_rolling_mean_60` | 60 periods | Long-term average |
| `requests_rolling_std_15` | 15 periods | Volatility measure |
| `requests_rolling_max_30` | 30 periods | Peak detection |
| `requests_rolling_min_30` | 30 periods | Trough detection |
| `bytes_rolling_mean_15` | 15 periods | Bandwidth trend |

**Code example:**
```python
df['requests_rolling_mean_5'] = df['request_count'].rolling(window=5).mean()
df['requests_rolling_std_15'] = df['request_count'].rolling(window=15).std()
```

### 3.5 Level 5: Advanced Features (Điểm cộng)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `requests_diff` | `requests(t) - requests(t-1)` | Trend direction |
| `requests_pct_change` | `(requests(t) - requests(t-1)) / requests(t-1)` | Growth rate |
| `spike_score` | `(x - rolling_mean) / rolling_std` | Anomaly detection |
| `host_entropy` | `-Σ(p_i * log(p_i))` | DDoS detection |
| `error_burst` | Consecutive errors | Overload indicator |
| `content_ratio` | HTML / (HTML + Images) | Traffic composition |
| `avg_response_size` | `total_bytes / request_count` | Size per request |

**Spike detection:**
```python
df['spike_score'] = (
    (df['request_count'] - df['requests_rolling_mean_15']) 
    / df['requests_rolling_std_15']
)
df['is_spike'] = (df['spike_score'] > 3).astype(int)  # 3-sigma rule
```

**Host entropy (DDoS detection):**
```python
from scipy.stats import entropy

def calculate_host_entropy(hosts):
    """High entropy = many different hosts (normal)
       Low entropy = few hosts dominating (possible DDoS)"""
    value_counts = hosts.value_counts()
    probabilities = value_counts / value_counts.sum()
    return entropy(probabilities)
```

### 3.6 Feature Matrix Summary

| Level | Features Count | Usage |
|-------|----------------|-------|
| Time features | 10 | All models |
| Aggregations | 9 (per window) | Target variables |
| Lag features | 6+ | ARIMA, LSTM |
| Rolling stats | 7+ | Tree-based, Neural Nets |
| Advanced | 7+ | Bonus points |
| **Total** | **~40-50 features** | Final dataset |

### 3.7 Target Variables

| Variable | Description | Use Case |
|----------|-------------|----------|
| `request_count` | Number of requests | Primary target for autoscaling |
| `bytes_sum` | Total bandwidth | Secondary target |
| `unique_hosts` | Active users | Capacity planning |
| `error_rate` | % of errors | Health monitoring |

---

## 4. KẾ HOẠCH THỰC HIỆN

### 4.1 Phase 1: Data Pipeline (3 ngày)

#### Day 1: Data Ingestion & Cleaning
**File: `notebooks/01_data_ingestion.ipynb`**

```python
# Tasks
1. Load train.txt and test.txt
2. Parse log với regex pattern
3. Extract 7 fields (host, timestamp, method, url, protocol, status, bytes)
4. Handle missing data (Aug 1-3 gap)
5. Data quality checks
   - Check parse success rate
   - Validate timestamp continuity
   - Identify outliers in bytes field
6. Save cleaned data → data/processed/cleaned_train.parquet
```

**Expected output:**
- `cleaned_train.parquet`: 2.9M rows × 7 columns
- `cleaned_test.parquet`: 526K rows × 7 columns
- Parse success rate: 100%

#### Day 2: Time Aggregation
**File: `notebooks/02_aggregation.ipynb`**

```python
# Tasks
1. Resample to 1-minute intervals
   - request_count_1m
   - bytes_sum_1m
   - unique_hosts_1m
   - error_rate_1m
   
2. Resample to 5-minute intervals
   - request_count_5m
   - bytes_sum_5m
   - avg_response_size_5m
   
3. Resample to 15-minute intervals
   - request_count_15m
   - bytes_sum_15m
   
4. Handle missing periods (fill with 0 or interpolate)
5. Save aggregated data
```

**Expected output:**
- `train_1m.parquet`: ~76,000 rows
- `train_5m.parquet`: ~15,200 rows
- `train_15m.parquet`: ~5,100 rows

#### Day 3: Feature Engineering
**File: `notebooks/03_feature_engineering.ipynb`**

```python
# Tasks
1. Add time features (hour, day_of_week, is_weekend, etc.)
2. Create lag features (1, 5, 12, 60, 288, 2016)
3. Calculate rolling statistics (mean, std, max, min)
4. Advanced features (spike_score, host_entropy)
5. Handle NaN values from lag/rolling (forward fill or drop)
6. Train/test split based on date
7. Save final feature sets
```

**Expected output:**
- `train_features_1m.parquet`: ~40 features
- `train_features_5m.parquet`: ~40 features
- `train_features_15m.parquet`: ~40 features
- Feature importance analysis

---

### 4.2 Phase 2: EDA & Insights (2 ngày)

#### Day 4: Exploratory Data Analysis
**File: `notebooks/04_eda.ipynb`**

**Visualizations to create:**

1. **Time Series Plots**
   ```python
   - Line plot: Daily request volume
   - Line plot: Hourly pattern (average by hour)
   - Heatmap: Day of week × Hour
   ```

2. **Seasonality Analysis**
   ```python
   from statsmodels.tsa.seasonal import seasonal_decompose
   - Decompose: Trend + Seasonal + Residual
   - ACF/PACF plots for ARIMA order selection
   ```

3. **Distribution Analysis**
   ```python
   - Histogram: Request counts
   - Box plot: Request by hour
   - Violin plot: Request by day of week
   ```

4. **Correlation Analysis**
   ```python
   - Correlation matrix heatmap
   - Feature importance (preliminary)
   ```

5. **Anomaly Detection Visual**
   ```python
   - Highlight spikes (Jul 13)
   - Mark missing data period (Aug 1-3)
   ```

**Expected output:**
- 15-20 visualizations
- Key insights document
- Recommended features for modeling

#### Day 5: Hypothesis & Feature Selection
**File: `notebooks/05_feature_selection.ipynb`**

```python
# Tasks
1. Test hypotheses:
   - Weekend traffic is significantly lower? (t-test)
   - Business hours have higher traffic? (ANOVA)
   - Is there weekly seasonality? (FFT analysis)
   
2. Feature selection:
   - Remove highly correlated features (> 0.95)
   - Feature importance from RandomForest
   - Mutual information scores
   
3. Prepare final feature sets for each model type
```

**Expected output:**
- Selected features list (25-30 features)
- Statistical test results
- Feature engineering insights

---

### 4.3 Phase 3: Modeling (4 ngày)

#### Day 6-7: Baseline Models
**File: `notebooks/06_baseline_models.ipynb`**

**Model 1: SARIMA**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Tasks
1. ACF/PACF analysis for order selection
2. Grid search for (p,d,q)(P,D,Q,m) parameters
3. Fit on train data (1m, 5m, 15m windows)
4. Forecast on test set
5. Evaluate: RMSE, MAE, MAPE
6. Plot: Predicted vs Actual
```

**Expected metrics (example):**
- RMSE (1m): ~15-20 requests
- RMSE (5m): ~50-80 requests
- RMSE (15m): ~150-250 requests

**Model 2: Prophet**
```python
from prophet import Prophet

# Tasks
1. Prepare data format (ds, y)
2. Add seasonality components:
   - Yearly: False (only 2 months data)
   - Weekly: True
   - Daily: True
3. Add holidays/events (Jul 13 spike?)
4. Fit and forecast
5. Evaluate metrics
6. Component plots
```

**Expected metrics:**
- RMSE (5m): ~40-60 requests
- MAPE: 15-25%

#### Day 8-9: Advanced Models
**File: `notebooks/07_ml_models.ipynb`**

**Time Series Cross-Validation Strategy:**
```python
from sklearn.model_selection import TimeSeriesSplit

# CRITICAL: Use expanding window CV (no data leakage)
tscv = TimeSeriesSplit(n_splits=5, gap=12)  # 12-period gap to prevent leakage

# Validation splits visualization:
# Fold 1: [Train: Day 1-10 ] | Gap | [Val: Day 11-15]
# Fold 2: [Train: Day 1-15 ] | Gap | [Val: Day 16-20]
# Fold 3: [Train: Day 1-20 ] | Gap | [Val: Day 21-25]
# ...
```

**Model 3: LightGBM**
```python
import lightgbm as lgb
import optuna

# Tasks
1. Prepare feature matrix (X) and target (y)
2. Time-based split validation (TimeSeriesSplit with gap)
3. Hyperparameter tuning with Optuna:
   - num_leaves: [31, 63, 127, 255]
   - learning_rate: [0.01, 0.03, 0.05, 0.1]
   - n_estimators: [100, 300, 500, 1000]
   - min_child_samples: [5, 10, 20]
   - subsample: [0.6, 0.8, 1.0]
   - colsample_bytree: [0.6, 0.8, 1.0]
4. Train on all features with early stopping
5. Feature importance analysis (SHAP values)
6. Evaluate on test set
7. Log experiments to MLflow
```

**Expected metrics:**
- RMSE (5m): ~30-45 requests (best)
- Feature importance: lag features + rolling stats top

**Model 4 (Optional): LSTM**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Tasks
1. Reshape data for LSTM (samples, timesteps, features)
2. Architecture:
   - LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(1)
3. Train with early stopping
4. Evaluate on test set
5. Visualize learning curves
```

**Expected metrics:**
- RMSE (5m): ~35-50 requests
- Training time: ~30-60 mins

#### Model Comparison Matrix

| Model | RMSE (5m) | MAE (5m) | MAPE | Training Time | Pros | Cons |
|-------|-----------|----------|------|---------------|------|------|
| SARIMA | ~60 | ~45 | 20% | 5 min | Interpretable, statistical | Slow, limited features |
| Prophet | ~50 | ~38 | 18% | 2 min | Easy, seasonality | Black box |
| LightGBM | ~35 | ~25 | 12% | 1 min | Fast, accurate | Needs features |
| LSTM | ~40 | ~30 | 15% | 45 min | Sequences | Slow, overfitting risk |

**Recommendation: LightGBM for production**

---

### 4.4 Phase 4: Autoscaling Logic (3 ngày)

#### Day 10: Scaling Policy Design
**File: `notebooks/08_scaling_policy.ipynb`**

**Scaling Parameters:**
```python
class ScalingConfig:
    # Capacity
    MIN_SERVERS = 1
    MAX_SERVERS = 20
    REQUESTS_PER_SERVER_PER_MIN = 100  # Capacity threshold
    
    # Thresholds
    SCALE_OUT_THRESHOLD = 0.80  # 80% capacity → add server
    SCALE_IN_THRESHOLD = 0.30   # 30% capacity → remove server
    
    # Timing
    SCALE_OUT_CONSECUTIVE = 5   # 5 minutes sustained high load
    SCALE_IN_CONSECUTIVE = 10   # 10 minutes sustained low load
    COOLDOWN_MINUTES = 5        # Wait 5 min between scaling actions
    
    # Hysteresis
    SCALE_OUT_INCREMENT = 2     # Add 2 servers at once
    SCALE_IN_DECREMENT = 1      # Remove 1 server at a time
```

**Scaling Algorithm:**
```python
def recommend_scaling(predicted_load, current_servers, history):
    """
    Args:
        predicted_load: List of predicted requests for next N minutes
        current_servers: Current number of active servers
        history: Recent scaling actions
    
    Returns:
        action: 'scale_out', 'scale_in', or 'hold'
        target_servers: Recommended server count
    """
    # 1. Calculate required servers
    required_servers = ceil(predicted_load.mean() / REQUESTS_PER_SERVER_PER_MIN)
    
    # 2. Calculate current utilization
    utilization = predicted_load.mean() / (current_servers * REQUESTS_PER_SERVER_PER_MIN)
    
    # 3. Check cooldown
    if last_action_time < COOLDOWN_MINUTES:
        return 'hold', current_servers
    
    # 4. Scale out logic
    if utilization > SCALE_OUT_THRESHOLD:
        consecutive_high = count_consecutive_high(predicted_load)
        if consecutive_high >= SCALE_OUT_CONSECUTIVE:
            target = min(current_servers + SCALE_OUT_INCREMENT, MAX_SERVERS)
            return 'scale_out', target
    
    # 5. Scale in logic
    elif utilization < SCALE_IN_THRESHOLD:
        consecutive_low = count_consecutive_low(predicted_load)
        if consecutive_low >= SCALE_IN_CONSECUTIVE:
            target = max(current_servers - SCALE_IN_DECREMENT, MIN_SERVERS)
            return 'scale_in', target
    
    # 6. Hold (do nothing)
    return 'hold', current_servers
```

#### Day 11: Simulation & Cost Analysis
**File: `notebooks/09_cost_simulation.ipynb`**

**Simulation:**
```python
# Simulate autoscaling on test set
results = []
current_servers = 1
scaling_events = []

for t, row in test_df.iterrows():
    # 1. Get prediction for next 15 minutes
    predicted = model.predict(row)
    
    # 2. Recommend scaling
    action, target = recommend_scaling(predicted, current_servers, scaling_events)
    
    # 3. Record metrics
    results.append({
        'timestamp': t,
        'actual_load': row['request_count'],
        'predicted_load': predicted,
        'servers': current_servers,
        'utilization': row['request_count'] / (current_servers * 100),
        'action': action
    })
    
    # 4. Execute action
    if action in ['scale_out', 'scale_in']:
        scaling_events.append({'time': t, 'action': action, 'from': current_servers, 'to': target})
        current_servers = target
```

**Cost Analysis:**
```python
# Pricing (ví dụ)
COST_PER_SERVER_PER_HOUR = 0.10  # $0.10/server/hour

# Strategy 1: Fixed (always max servers)
fixed_cost = MAX_SERVERS * COST_PER_SERVER_PER_HOUR * total_hours

# Strategy 2: Autoscaling
autoscale_cost = sum(servers_at_time * COST_PER_SERVER_PER_HOUR * (1/60)) for each minute

# Strategy 3: Fixed minimal (always min servers)
minimal_cost = MIN_SERVERS * COST_PER_SERVER_PER_HOUR * total_hours

# Performance metrics
sla_violations = count(utilization > 1.0)  # Overloaded periods
wasted_capacity = count(utilization < 0.3)  # Underutilized periods
```

**Expected results:**
- Autoscaling cost: ~40-60% reduction vs. fixed max
- SLA violations: < 1% of time
- Avg utilization: 60-80%

#### Day 12: Optimization & Tuning
**File: `notebooks/10_policy_optimization.ipynb`**

```python
# Tasks
1. Grid search optimal thresholds:
   - SCALE_OUT_THRESHOLD: [0.70, 0.75, 0.80, 0.85]
   - SCALE_IN_THRESHOLD: [0.20, 0.25, 0.30, 0.35]
   
2. Test different cooldown periods: [3, 5, 7, 10] minutes

3. Evaluate trade-offs:
   - Cost vs. SLA compliance
   - Responsiveness vs. stability (flapping)
   
4. Visualize:
   - Cost-performance frontier
   - Scaling timeline with events
   - Utilization heatmap
```

---

### 4.5 Phase 5: Deployment (3 ngày)

#### Day 13-14: API Development
**File: `src/api/main.py`**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Autoscaling Prediction API")

# Load models
model_5m = joblib.load('models/lgbm_5m.pkl')
scaler = joblib.load('models/scaler.pkl')

# Endpoints
@app.post("/forecast")
async def forecast(horizon: int = 30, confidence: float = 0.95):
    """Predict request volume for next N minutes with prediction intervals"""
    predictions = model_5m.predict(horizon)
    lower, upper = model_5m.predict_interval(horizon, confidence)
    return {
        "horizon_minutes": horizon,
        "predictions": predictions.tolist(),
        "prediction_lower": lower.tolist(),  # Lower bound
        "prediction_upper": upper.tolist(),  # Upper bound
        "confidence_level": confidence,
        "mean_load": float(predictions.mean()),
        "peak_load": float(predictions.max())
    }

@app.post("/recommend-scaling")
async def recommend_scaling(predicted_load: list, current_servers: int):
    """Get scaling recommendation"""
    action, target = scaling_policy.recommend(predicted_load, current_servers)
    return {
        "action": action,
        "target_servers": target,
        "current_servers": current_servers,
        "estimated_utilization": calculate_utilization(predicted_load, target)
    }

@app.get("/metrics")
async def get_metrics():
    """Current system metrics"""
    return {
        "model_rmse": 35.2,
        "model_mae": 25.1,
        "avg_servers": 3.4,
        "cost_reduction": "45%"
    }

@app.get("/cost-report")
async def cost_report(start_date: str, end_date: str):
    """Cost analysis for date range"""
    return {
        "period": f"{start_date} to {end_date}",
        "total_cost": 123.45,
        "fixed_cost_comparison": 234.56,
        "savings": 111.11,
        "scaling_events": 42
    }
```

**Testing:**
```bash
# Start server
uvicorn src.api.main:app --reload --port 8000

# Test endpoints
curl -X POST "http://localhost:8000/forecast?horizon=30"
curl -X POST "http://localhost:8000/recommend-scaling" \
  -H "Content-Type: application/json" \
  -d '{"predicted_load": [120, 150, 180], "current_servers": 2}'
```

#### Day 15: Dashboard Development
**File: `app/dashboard.py`**

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Autoscaling Dashboard", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Configuration")
time_window = st.sidebar.selectbox("Time Window", ["1 minute", "5 minutes", "15 minutes"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (min)", 5, 60, 30)

# Main dashboard
st.title("🚀 Autoscaling Analysis Dashboard")

# Row 1: Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Load", "145 req/min", "+12%")
with col2:
    st.metric("Active Servers", "3", "↑1")
with col3:
    st.metric("Utilization", "75%", "+5%")
with col4:
    st.metric("Est. Cost/Hour", "$0.30", "-$0.15")

# Row 2: Time Series
st.subheader("📈 Traffic & Forecast")
col1, col2 = st.columns([2, 1])

with col1:
    # Historical + Forecast plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['actual'], name='Actual', mode='lines'))
    fig.add_trace(go.Scatter(x=df_pred['timestamp'], y=df_pred['predicted'], name='Forecast', mode='lines', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['servers']*100, name='Capacity', mode='lines', line=dict(color='green')))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Scaling recommendation
    st.subheader("🎯 Recommendation")
    prediction = requests.post(f"http://localhost:8000/forecast?horizon={forecast_horizon}").json()
    recommendation = requests.post("http://localhost:8000/recommend-scaling", 
                                   json={"predicted_load": prediction['predictions'], "current_servers": 3}).json()
    
    if recommendation['action'] == 'scale_out':
        st.error(f"⬆️ SCALE OUT to {recommendation['target_servers']} servers")
    elif recommendation['action'] == 'scale_in':
        st.info(f"⬇️ SCALE IN to {recommendation['target_servers']} servers")
    else:
        st.success("✅ HOLD - Current capacity adequate")
    
    st.metric("Predicted Avg Load", f"{prediction['mean_load']:.0f} req/min")
    st.metric("Predicted Peak", f"{prediction['peak_load']:.0f} req/min")

# Row 3: Analysis
tab1, tab2, tab3 = st.tabs(["📊 Patterns", "💰 Cost Analysis", "⚠️ Anomalies"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        # Hourly pattern
        fig = px.box(df, x='hour', y='request_count', title='Request Volume by Hour')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Day of week pattern
        fig = px.box(df, x='day_of_week', y='request_count', title='Request Volume by Day of Week')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Cost comparison
    cost_data = pd.DataFrame({
        'Strategy': ['Fixed (Max)', 'Autoscaling', 'Fixed (Min)'],
        'Cost': [234.56, 123.45, 67.89],
        'SLA Violations': [0, 5, 342]
    })
    fig = px.bar(cost_data, x='Strategy', y='Cost', title='Cost Comparison')
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("Cost Savings", "$111.11", "-47%")
    st.metric("SLA Compliance", "99.5%", "+0.5%")

with tab3:
    # Anomaly detection
    anomalies = df[df['spike_score'] > 3]
    st.dataframe(anomalies[['timestamp', 'request_count', 'spike_score']])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['request_count'], name='Normal', mode='markers', marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['request_count'], name='Anomaly', mode='markers', marker=dict(size=10, color='red')))
    st.plotly_chart(fig, use_container_width=True)
```

**Running:**
```bash
streamlit run app/dashboard.py
```

---

### 4.6 Phase 6: Documentation (2 ngày)

#### Day 16: Report & README
**Files: `reports/report.pdf`, `README.md`**

**Report Structure (max 30 pages):**
1. Executive Summary (1 page)
2. Problem Statement (2 pages)
3. Data Analysis (5 pages)
   - Data overview
   - EDA insights
   - Feature engineering
4. Methodology (8 pages)
   - Model selection rationale
   - Training process
   - Hyperparameter tuning
5. Results (6 pages)
   - Model comparison
   - Performance metrics
   - Scaling policy evaluation
6. Deployment (3 pages)
   - Architecture
   - API documentation
   - Dashboard features
7. Conclusion & Future Work (2 pages)
8. References (1 page)
9. Appendix (2 pages)
   - Code snippets
   - Additional visualizations

**README.md (follow sample-README.md template):**
- Project overview
- Installation instructions
- Usage guide
- API endpoints
- Model performance
- Team info

#### Day 17: Slides & Video
**Files: `reports/slides.pptx`, `demo_video.mp4`**

**Slide Structure (15-20 slides):**
1. Title & Team (1)
2. Problem Introduction (2)
3. Data Overview (2)
4. Key Insights from EDA (3)
5. Modeling Approach (3)
6. Results & Comparison (3)
7. Autoscaling Logic (2)
8. Demo (2)
9. Cost Analysis (1)
10. Conclusion & Q&A (1)

**Video Demo Script (3-5 minutes):**
```
[0:00-0:30] Introduction & Problem
[0:30-1:00] Data visualization highlights
[1:00-2:00] Live API demo
  - Call /forecast endpoint
  - Show prediction results
  - Call /recommend-scaling
[2:00-3:30] Dashboard walkthrough
  - Real-time metrics
  - Forecast visualization
  - Scaling recommendations
  - Cost analysis tab
[3:30-4:00] Key results & benefits
[4:00-4:30] Q&A preparation
```

---

## 5. KIẾN TRÚC HỆ THỐNG

### 5.1 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Raw Logs (train.txt, test.txt)                                 │
│       ↓                                                          │
│  Parser (regex extraction)                                       │
│       ↓                                                          │
│  Cleaned Data (Parquet)                                          │
│       ↓                                                          │
│  Aggregator (1m, 5m, 15m windows)                               │
│       ↓                                                          │
│  Feature Engineer (40+ features)                                 │
│       ↓                                                          │
│  Final Datasets (train_features_*.parquet)                       │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MODELING LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────┐             │
│  │ SARIMA  │  │ Prophet │  │ LightGBM │  │ LSTM │             │
│  └────┬────┘  └────┬────┘  └─────┬────┘  └───┬──┘             │
│       └────────────┴─────────────┴───────────┘                  │
│                         ↓                                        │
│              Model Selection (LightGBM)                          │
│                         ↓                                        │
│             Saved Model (.pkl, .joblib)                          │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐              ┌────────────────────┐       │
│  │   FastAPI        │              │   Streamlit        │       │
│  │   (Backend)      │◄────────────►│   (Frontend)       │       │
│  │                  │    REST API  │                    │       │
│  │  /forecast       │              │  📈 Visualizations │       │
│  │  /recommend      │              │  🎛️ Controls       │       │
│  │  /metrics        │              │  📊 Analytics      │       │
│  │  /cost-report    │              │  💰 Cost Dashboard │       │
│  └──────────────────┘              └────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SCALING LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Autoscaling Policy Engine                                       │
│    ↓                                                             │
│  Simulator (Cost & Performance Analysis)                         │
│    ↓                                                             │
│  Recommendations (scale_out / scale_in / hold)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Directory Structure (Final)

```
datafollow/
├── DATA/
│   ├── train.txt                      # Raw log (304 MB)
│   ├── test.txt                       # Raw log (54 MB)
│   ├── processed/
│   │   ├── cleaned_train.parquet      # Cleaned raw data
│   │   ├── cleaned_test.parquet
│   │   ├── train_1m.parquet           # Aggregated 1-min
│   │   ├── train_5m.parquet           # Aggregated 5-min
│   │   ├── train_15m.parquet          # Aggregated 15-min
│   │   ├── train_features_1m.parquet  # Final features
│   │   ├── train_features_5m.parquet
│   │   └── train_features_15m.parquet
│   └── sample-README.md               # Template
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb        # Parse & clean
│   ├── 02_aggregation.ipynb           # Time windows
│   ├── 03_feature_engineering.ipynb   # Features
│   ├── 04_eda.ipynb                   # Visualizations
│   ├── 05_feature_selection.ipynb     # Feature importance
│   ├── 06_baseline_models.ipynb       # SARIMA, Prophet
│   ├── 07_ml_models.ipynb             # LightGBM, LSTM
│   ├── 08_scaling_policy.ipynb        # Policy design
│   ├── 09_cost_simulation.ipynb       # Simulation
│   └── 10_policy_optimization.ipynb   # Tuning
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parser.py                  # Log parsing logic
│   │   ├── cleaner.py                 # Data cleaning
│   │   └── aggregator.py              # Time aggregation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── time_features.py           # Hour, day, etc.
│   │   ├── lag_features.py            # Lag generation
│   │   ├── rolling_features.py        # Rolling stats
│   │   └── advanced_features.py       # Spike, entropy
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sarima.py                  # SARIMA wrapper
│   │   ├── prophet_model.py           # Prophet wrapper
│   │   ├── lgbm_model.py              # LightGBM wrapper
│   │   └── lstm_model.py              # LSTM implementation
│   ├── scaling/
│   │   ├── __init__.py
│   │   ├── config.py                  # Scaling parameters
│   │   ├── policy.py                  # Scaling logic
│   │   └── simulator.py               # Cost simulation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # RMSE, MAE, MAPE
│   │   └── visualization.py           # Plot helpers
│   └── api/
│       ├── __init__.py
│       └── main.py                    # FastAPI app
│
├── app/
│   └── dashboard.py                   # Streamlit dashboard
│
├── models/                            # Saved models
│   ├── sarima_5m.pkl
│   ├── prophet_5m.pkl
│   ├── lgbm_5m.pkl
│   ├── lgbm_1m.pkl
│   ├── scaler.pkl
│   └── feature_names.json
│
├── reports/
│   ├── report.pdf                     # Final report (max 30 pages)
│   ├── slides.pptx                    # Presentation
│   ├── demo_video.mp4                 # 3-5 min video
│   └── figures/                       # All visualizations
│
├── tests/                             # Unit tests (required)
│   ├── test_parser.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_scaling_policy.py
│
├── mlruns/                            # MLflow experiment tracking
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Run tests on PR
│       └── cd.yml                     # Deploy pipeline
│
├── .gitignore
├── .pre-commit-config.yaml            # Pre-commit hooks
├── Dockerfile                         # API container
├── Dockerfile.streamlit               # Dashboard container
├── docker-compose.yml                 # Multi-container setup
├── dvc.yaml                           # DVC pipeline
├── pyproject.toml                     # Project config (replaces setup.py)
├── requirements.txt
├── README.md                          # Main documentation
└── PROJECT_PLAN.md                    # This file
```

### 5.3 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Processing** | Pandas, Polars, NumPy | Data manipulation |
| **Time Series** | Statsmodels, Prophet, sktime | SARIMA, forecasting |
| **ML Models** | LightGBM, XGBoost, Scikit-learn | Gradient boosting |
| **Deep Learning** | PyTorch, Lightning | LSTM |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization |
| **Visualization** | Matplotlib, Seaborn, Plotly | Charts & plots |
| **API** | FastAPI | REST endpoints |
| **Dashboard** | Streamlit | Interactive UI |
| **Storage** | Parquet (PyArrow) | Efficient data format |
| **MLOps** | MLflow, DVC | Experiment tracking |
| **Containerization** | Docker, Docker Compose | Deployment |
| **Version Control** | Git | Code management |
| **Code Quality** | Ruff, Black, pre-commit | Linting, formatting |

### 5.3.1 MLOps Pipeline

```
Experiment Tracking (MLflow):
├── Log parameters, metrics, artifacts
├── Model registry (staging -> production)
└── Compare runs across experiments

Data Versioning (DVC):
├── Track large data files (train.txt, test.txt)
├── Reproducible pipelines
└── Remote storage (S3/GCS compatible)
```

### 5.3.2 Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - api
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
```

### 5.3.3 Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/ -v --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

### 5.4 Dependencies (requirements.txt)

```txt
# Core data processing
pandas>=2.2.0
numpy>=1.26.0
pyarrow>=15.0.0
polars>=0.20.0  # Fast alternative for large data

# Time series & forecasting
statsmodels>=0.14.1
prophet>=1.1.5
sktime>=0.26.0  # Unified time series framework

# Machine learning
scikit-learn>=1.4.0
lightgbm>=4.3.0
xgboost>=2.0.3
optuna>=3.5.0  # Hyperparameter optimization

# Deep learning (optional)
torch>=2.2.0  # PyTorch for LSTM (lighter than TensorFlow)
lightning>=2.2.0  # PyTorch Lightning for training

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# Web frameworks
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
streamlit>=1.31.0

# MLOps & Experiment Tracking
mlflow>=2.10.0  # Experiment tracking
dvc>=3.42.0  # Data version control

# Utilities
python-dateutil>=2.8.2
scipy>=1.12.0
joblib>=1.3.2
pydantic>=2.6.0

# Development & Testing
jupyter>=1.0.0
notebook>=7.1.0
ipykernel>=6.29.0
pytest>=8.0.0
pytest-cov>=4.1.0  # Coverage reporting
black>=24.1.0
ruff>=0.2.0  # Fast linter
pre-commit>=3.6.0  # Git hooks

# Containerization
docker>=7.0.0  # Docker SDK
```

---

## 6. TIMELINE & CHECKLIST

### 6.1 Gantt Chart Overview

```
Week 1: Data Pipeline & EDA
├── Day 1  : ■■■■ Data Ingestion + DVC Setup
├── Day 2  : ■■■■ Aggregation
├── Day 3  : ■■■■ Feature Engineering
├── Day 4  : ■■■■ EDA
└── Day 5  : ■■■■ Feature Selection

Week 2: Modeling + MLOps
├── Day 6  : ■■■■ MLflow Setup + SARIMA
├── Day 7  : ■■■■ Prophet
├── Day 8  : ■■■■ LightGBM + Optuna
└── Day 9  : ■■■■ LSTM + Model Comparison

Week 3: Autoscaling & Deployment
├── Day 10 : ■■■■ Scaling Policy
├── Day 11 : ■■■■ Cost Simulation
├── Day 12 : ■■■■ Policy Optimization
├── Day 13 : ■■■■ API Development + Docker
└── Day 14 : ■■■■ Dashboard Development

Week 4: Testing, CI/CD & Documentation
├── Day 15 : ■■■■ Unit Tests + CI/CD Pipeline
├── Day 16 : ■■■■ Integration Tests + Docker Compose
├── Day 17 : ■■■■ Report Writing
├── Day 18 : ■■■■ Slides & Video
└── Day 19 : ■■■■ Final Review & Submission
```

### 6.2 Detailed Checklist

#### ✅ Phase 1: Data Pipeline
- [ ] Load train.txt and test.txt successfully
- [ ] Parse all records with regex (100% success rate)
- [ ] Extract 7 fields correctly
- [ ] Handle missing data period (Aug 1-3)
- [ ] Create 1-minute aggregation
- [ ] Create 5-minute aggregation
- [ ] Create 15-minute aggregation
- [ ] Generate time features (hour, day_of_week, etc.)
- [ ] Create lag features (1, 5, 12, 60, 288, 2016)
- [ ] Calculate rolling statistics (mean, std, max, min)
- [ ] Implement advanced features (spike_score, host_entropy)
- [ ] Save all processed datasets to Parquet

#### ✅ Phase 2: EDA
- [ ] Create time series line plots
- [ ] Generate hourly pattern visualization
- [ ] Create day-of-week heatmap
- [ ] Perform seasonality decomposition
- [ ] Plot ACF/PACF for ARIMA
- [ ] Create correlation matrix
- [ ] Visualize distribution (histogram, box plots)
- [ ] Identify and mark anomalies (Jul 13 spike)
- [ ] Statistical hypothesis tests (weekend effect, business hours)
- [ ] Feature importance preliminary analysis
- [ ] Document key insights

#### ✅ Phase 3: Modeling
- [ ] Set up MLflow experiment tracking
- [ ] Implement TimeSeriesSplit cross-validation with gap
- [ ] Implement SARIMA model
- [ ] Tune SARIMA hyperparameters (p,d,q)(P,D,Q,m)
- [ ] Evaluate SARIMA: RMSE, MAE, MAPE
- [ ] Implement Prophet model
- [ ] Add seasonality components to Prophet
- [ ] Evaluate Prophet metrics
- [ ] Implement LightGBM model
- [ ] Hyperparameter tuning with Optuna (Bayesian)
- [ ] Feature importance analysis (SHAP values)
- [ ] Evaluate LightGBM metrics
- [ ] (Optional) Implement LSTM with PyTorch Lightning
- [ ] Compare all models (create comparison table)
- [ ] Select best model for production
- [ ] Register model in MLflow Model Registry
- [ ] Save trained models with versioning

#### ✅ Phase 4: Autoscaling
- [ ] Define scaling configuration (thresholds, cooldown, etc.)
- [ ] Implement scaling recommendation function
- [ ] Add cooldown mechanism
- [ ] Implement hysteresis logic
- [ ] Simulate autoscaling on test set
- [ ] Calculate cost for different strategies
- [ ] Compare cost: Fixed vs. Autoscaling vs. Minimal
- [ ] Measure SLA violations
- [ ] Optimize thresholds (grid search)
- [ ] Visualize scaling timeline
- [ ] Create cost-performance frontier plot
- [ ] Document optimal configuration

#### ✅ Phase 5: Deployment
- [ ] Create FastAPI application structure
- [ ] Implement /forecast endpoint
- [ ] Implement /recommend-scaling endpoint
- [ ] Implement /metrics endpoint
- [ ] Implement /cost-report endpoint
- [ ] Test all API endpoints
- [ ] Write API documentation (OpenAPI/Swagger)
- [ ] Create Streamlit dashboard layout
- [ ] Add real-time metrics display
- [ ] Implement forecast visualization
- [ ] Add scaling recommendation widget
- [ ] Create cost analysis tab
- [ ] Add anomaly detection tab
- [ ] Test dashboard functionality
- [ ] Connect dashboard to API

#### ✅ Phase 6: Documentation
- [ ] Write executive summary
- [ ] Document data analysis section
- [ ] Document methodology
- [ ] Create results section with tables/charts
- [ ] Write deployment section
- [ ] Add conclusion & future work
- [ ] Complete references
- [ ] Create appendix
- [ ] Proofread entire report (max 30 pages)
- [ ] Write comprehensive README.md
- [ ] Create presentation slides (15-20 slides)
- [ ] Record demo video (3-5 minutes)
- [ ] Edit and finalize video
- [ ] Prepare Q&A talking points

#### ✅ Phase 7: MLOps & CI/CD
- [ ] Set up pre-commit hooks (black, ruff)
- [ ] Configure DVC for data versioning
- [ ] Write unit tests (80% coverage minimum)
- [ ] Create GitHub Actions CI workflow
- [ ] Build Docker images for API and Dashboard
- [ ] Test docker-compose deployment locally
- [ ] Set up MLflow Model Registry workflow
- [ ] Document deployment process

#### ✅ Final Submission Checklist
- [ ] GitHub repo is public and accessible
- [ ] README.md is complete with installation guide
- [ ] All code runs without errors
- [ ] All tests pass (80%+ coverage)
- [ ] Docker containers build and run successfully
- [ ] Models are saved and documented in MLflow
- [ ] API is functional (tested with curl/Postman)
- [ ] Dashboard is functional
- [ ] Report PDF (max 30 pages)
- [ ] Presentation slides (PPTX)
- [ ] Demo video (3-5 min, MP4)
- [ ] Link GitHub repo in report
- [ ] No commits after submission deadline
- [ ] All team members reviewed final submission

---

## 7. KẾT LUẬN & LƯU Ý

### 7.1 Key Success Factors

1. **Data Quality**: 100% parse success rate, proper handling of missing period
2. **Feature Engineering**: Comprehensive set of 40+ features tailored for time series
3. **Model Selection**: Balanced approach with statistical + ML models
4. **Practical Scaling**: Realistic policy with cooldown and hysteresis
5. **Clear Documentation**: Well-structured report and working demo

### 7.2 Potential Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Missing data (Aug 1-3) | Forward fill or exclude from training |
| Model overfitting | Cross-validation with time-based split |
| Lag feature NaN | Drop initial rows or forward fill |
| API performance | Cache predictions, async processing |
| Dashboard responsiveness | Sample data for large plots |
| Cost estimation accuracy | Validate with real cloud pricing |

### 7.3 Bonus Points Opportunities

- ✨ **Anomaly Detection**: Implement 3-sigma spike detection + host entropy
- ✨ **Smart Cooldown**: Adaptive cooldown based on load volatility
- ✨ **Cost Optimization**: Multi-objective optimization (cost vs. performance)
- ✨ **Real-time Simulation**: Interactive slider in dashboard to test policies
- ✨ **DDoS Detection**: Low host entropy alert system
- ✨ **Uncertainty Quantification**: Prediction intervals using:
  - Quantile regression with LightGBM (`objective='quantile'`)
  - Conformal prediction for distribution-free intervals
  - Prophet's built-in uncertainty
- ✨ **SHAP Explanations**: Model interpretability with SHAP values
- ✨ **MLOps Pipeline**: Full MLflow + DVC + Docker + CI/CD setup
- ✨ **Proactive Scaling**: Predict spikes 15-30 min ahead, scale before load hits

### 7.4 Next Steps

1. **Start**: Begin with Phase 1 - Data Ingestion (Day 1)
2. **Track Progress**: Update this checklist daily
3. **Daily Stand-up**: Review completed tasks and blockers
4. **Git Commits**: Commit code at end of each phase
5. **Peer Review**: Cross-check notebooks and code quality
6. **Documentation**: Write README and comments as you code (not at the end!)

---

**Last Updated**: January 25, 2026
**Version**: 2.0
**Status**: Ready to Execute

**Changes in v2.0**:
- Added MLOps pipeline (MLflow, DVC)
- Updated dependencies to latest versions
- Added Docker containerization
- Added TimeSeriesSplit with gap for validation
- Added Optuna for hyperparameter tuning
- Added prediction intervals to forecast endpoint
- Added CI/CD with GitHub Actions
- Added SHAP for model interpretability
- Extended timeline to 19 days for testing/CI

---

## 📞 CONTACT & RESOURCES

### Useful Resources
- NASA Log Dataset: http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
- Prophet Documentation: https://facebook.github.io/prophet/
- LightGBM Documentation: https://lightgbm.readthedocs.io/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Streamlit Documentation: https://docs.streamlit.io/

### Competition Info
- Website: https://dataflow.hamictoantin.com/vi
- Fanpage: https://www.facebook.com/toantinhamic
- Email: hamic@hus.edu.vn

---

**Good luck! 🍀**
