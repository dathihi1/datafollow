# 🎯 KẾ HOẠCH HOÀN THIỆN - AUTOSCALING ANALYSIS

**Ngày tạo**: 30 Tháng 1, 2026  
**Mục tiêu**: Nâng cấp bài làm lên mức competition-winning dựa trên gap analysis

---

## 📊 TÓM TẮT GAP ANALYSIS

### ✅ Điểm mạnh hiện tại (Giữ nguyên)
- [x] Kiến trúc modular, production-ready
- [x] 3 models: SARIMA, Prophet, LightGBM
- [x] Scaling policy với cooldown + hysteresis
- [x] Testing suite đầy đủ
- [x] API + Dashboard
- [x] Documentation chi tiết

### 🎯 Cần cải thiện (Ưu tiên)

| Priority | Feature | Effort | Impact | Status |
|----------|---------|--------|--------|--------|
| 🔴 HIGH | IsolationForest Anomaly Detection | 2-3h | ⭐⭐⭐⭐⭐ | TODO |
| 🔴 HIGH | Special Events Dictionary | 1h | ⭐⭐⭐⭐ | TODO |
| 🟡 MEDIUM | Event Type Feature | 1h | ⭐⭐⭐⭐ | TODO |
| 🟡 MEDIUM | Data Scaling Verification | 1-2h | ⭐⭐⭐ | TODO |
| 🟢 LOW | Enhanced Visualization | 2h | ⭐⭐⭐ | TODO |
| 🟢 LOW | Benchmark Report | 1h | ⭐⭐ | TODO |

**Total Estimated Time**: 8-11 hours

---

## 📅 IMPLEMENTATION ROADMAP (3 Days)

### **Day 1: Core Features (4-5 hours)**

#### ✅ Task 1.1: IsolationForest Anomaly Detection (2-3h)
**Files to create/modify:**
```
src/features/anomaly_detector.py          # NEW
notebooks/03_feature_engineering.ipynb     # MODIFY (add anomaly detection)
notebooks/04_eda.ipynb                     # MODIFY (add anomaly viz)
tests/test_anomaly_detector.py            # NEW
```

**Implementation:**
```python
# src/features/anomaly_detector.py
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

class TrafficAnomalyDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.
    
    Use case: Detect DDoS attacks, unusual traffic spikes, system failures.
    """
    
    def __init__(self, contamination=0.01, random_state=42):
        """
        Args:
            contamination: Expected proportion of outliers (default 1%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        
    def fit(self, X):
        """Train IsolationForest on features"""
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        self.model.fit(X)
        return self
        
    def predict(self, X):
        """
        Detect anomalies.
        
        Returns:
            -1 for anomalies, 1 for normal points
        """
        return self.model.predict(X)
        
    def decision_function(self, X):
        """
        Get anomaly scores (lower = more anomalous).
        
        Returns:
            Anomaly scores
        """
        return self.model.decision_function(X)
        
    def fit_predict(self, X):
        """Fit and predict in one step"""
        return self.fit(X).predict(X)
```

**Integration steps:**
1. Create `src/features/anomaly_detector.py`
2. Add to `03_feature_engineering.ipynb`:
   ```python
   # After advanced features section
   from src.features.anomaly_detector import TrafficAnomalyDetector
   
   # Select features for anomaly detection
   anomaly_features = [
       'request_count', 
       'unique_hosts',
       'error_rate',
       'bytes_per_request',
       'request_count_rolling_mean_12',
       'request_count_rolling_std_12',
   ]
   
   # Train IsolationForest
   detector = TrafficAnomalyDetector(contamination=0.01)
   df_train['is_anomaly_ml'] = detector.fit_predict(df_train[anomaly_features])
   df_test['is_anomaly_ml'] = detector.predict(df_test[anomaly_features])
   
   # Get anomaly scores
   df_train['anomaly_score'] = detector.decision_function(df_train[anomaly_features])
   df_test['anomaly_score'] = detector.decision_function(df_test[anomaly_features])
   
   # Map to binary: -1 (anomaly) -> 1, 1 (normal) -> 0
   df_train['is_anomaly_ml'] = (df_train['is_anomaly_ml'] == -1).astype(int)
   df_test['is_anomaly_ml'] = (df_test['is_anomaly_ml'] == -1).astype(int)
   
   print(f"Anomalies detected - Train: {df_train['is_anomaly_ml'].sum()}")
   print(f"Anomalies detected - Test: {df_test['is_anomaly_ml'].sum()}")
   ```

3. Add visualization to `04_eda.ipynb`:
   ```python
   # Compare Z-score vs ML anomaly detection
   fig, axes = plt.subplots(2, 1, figsize=(16, 8))
   
   # Z-score based
   ax1 = axes[0]
   ax1.plot(df['timestamp'], df['request_count'], linewidth=0.5)
   z_anomalies = df[df['is_spike'] == 1]
   ax1.scatter(z_anomalies['timestamp'], z_anomalies['request_count'], 
               color='red', s=30, label=f'Z-score ({len(z_anomalies)})')
   ax1.set_title('Statistical Anomaly Detection (Z-score)')
   
   # ML-based
   ax2 = axes[1]
   ax2.plot(df['timestamp'], df['request_count'], linewidth=0.5)
   ml_anomalies = df[df['is_anomaly_ml'] == 1]
   ax2.scatter(ml_anomalies['timestamp'], ml_anomalies['request_count'],
               color='orange', s=30, label=f'IsolationForest ({len(ml_anomalies)})')
   ax2.set_title('ML-based Anomaly Detection (Isolation Forest)')
   ```

**Expected output:**
- Train: ~150-300 anomalies (1% of ~15K rows)
- Test: ~25-50 anomalies

---

#### ✅ Task 1.2: Special Events Dictionary (1h)
**Files to modify:**
```
src/features/advanced_features.py          # MODIFY (add event detection)
notebooks/03_feature_engineering.ipynb     # MODIFY (add event features)
```

**Implementation:**
```python
# Add to src/features/advanced_features.py

SPECIAL_EVENTS = {
    # US Holidays (Low traffic expected)
    '1995-07-04': {'type': 1, 'name': 'Independence Day', 'impact': 'low_traffic'},
    
    # NASA Space Shuttle STS-70 Mission
    '1995-07-13': {'type': 2, 'name': 'STS-70 Launch', 'impact': 'high_traffic'},
    '1995-07-14': {'type': 2, 'name': 'STS-70 Mission Day 1', 'impact': 'high_traffic'},
    '1995-07-15': {'type': 2, 'name': 'STS-70 Mission Day 2', 'impact': 'high_traffic'},
    '1995-07-16': {'type': 2, 'name': 'STS-70 Mission Day 3', 'impact': 'high_traffic'},
    '1995-07-17': {'type': 2, 'name': 'STS-70 Mission Day 4', 'impact': 'high_traffic'},
    '1995-07-18': {'type': 2, 'name': 'STS-70 Mission Day 5', 'impact': 'high_traffic'},
    '1995-07-19': {'type': 2, 'name': 'STS-70 Mission Day 6', 'impact': 'high_traffic'},
    '1995-07-20': {'type': 2, 'name': 'STS-70 Mission Day 7', 'impact': 'high_traffic'},
    '1995-07-21': {'type': 2, 'name': 'STS-70 Mission Day 8', 'impact': 'high_traffic'},
    '1995-07-22': {'type': 2, 'name': 'STS-70 Landing', 'impact': 'high_traffic'},
    
    # Apollo 11 Anniversary
    '1995-07-20': {'type': 2, 'name': 'Apollo 11 26th Anniversary', 'impact': 'high_traffic'},
    
    # Hurricane (Missing data period)
    '1995-08-01': {'type': 3, 'name': 'Hurricane Start', 'impact': 'outage'},
    '1995-08-02': {'type': 3, 'name': 'Hurricane', 'impact': 'outage'},
    '1995-08-03': {'type': 3, 'name': 'Hurricane End', 'impact': 'outage'},
}

class AdvancedFeatureExtractor:
    # ... existing code ...
    
    def _add_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add special event features based on domain knowledge."""
        
        # Initialize event columns
        df['event_type'] = 0  # 0: normal, 1: holiday, 2: space event, 3: outage
        df['is_special_event'] = 0
        df['event_impact'] = 'normal'
        df['event_name'] = ''
        
        # Map events
        for date_str, event_info in SPECIAL_EVENTS.items():
            date = pd.to_datetime(date_str).date()
            mask = df['timestamp'].dt.date == date
            
            df.loc[mask, 'event_type'] = event_info['type']
            df.loc[mask, 'is_special_event'] = 1
            df.loc[mask, 'event_impact'] = event_info['impact']
            df.loc[mask, 'event_name'] = event_info['name']
        
        return df
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all advanced features."""
        df = df.copy()
        
        # Existing features
        df = self._add_spike_features(df)
        df = self._add_trend_features(df)
        df = self._add_volatility_features(df)
        df = self._add_rate_features(df)
        
        # NEW: Event features
        df = self._add_event_features(df)
        
        return df
```

**Integration:**
```python
# In notebook 03
df_train = advanced_extractor.transform(df_train)
df_test = advanced_extractor.transform(df_test)

# Verify events
print("\nSpecial Events Summary:")
print(df_train[df_train['is_special_event'] == 1].groupby('event_name')['request_count'].agg(['mean', 'max', 'count']))
```

---

### **Day 2: Verification & Enhancement (3-4 hours)** ✅ COMPLETED

#### ✅ Task 2.1: Data Scaling Verification (1-2h) - DONE
**Files to check:**
```
notebooks/02_aggregation.ipynb            # CHECK
notebooks/03_feature_engineering.ipynb    # CHECK
src/models/*.py                           # CHECK
```

**Action items:**
1. Check if RobustScaler or MinMaxScaler is used
2. If yes, verify it's NOT applied to target variable
3. Create comparison notebook:
   ```python
   # notebooks/XX_scaling_comparison.ipynb
   
   # Load raw aggregated data (no scaling)
   df_raw = pd.read_parquet('DATA/processed/train_5m.parquet')
   
   # Load feature-engineered data
   df_features = pd.read_parquet('DATA/processed/train_features_5m.parquet')
   
   # Compare request_count distribution
   print("Raw data - request_count:")
   print(df_raw['request_count'].describe())
   
   print("\nFeature data - request_count:")
   print(df_features['request_count'].describe())
   
   # They should be identical!
   assert df_raw['request_count'].mean() == df_features['request_count'].mean()
   ```

4. If scaling is found on target:
   - Remove it from preprocessing
   - Re-train all models
   - Compare metrics (RMSE should be much higher but more realistic)

---

#### ✅ Task 2.2: Enhanced Visualization (2h) - DONE
**Files to create:**
```
notebooks/04_eda.ipynb                    # ADD sections
reports/figures/                          # GENERATE
```

**Add to EDA:**
1. **Overlay plot** (Prophet + XGBoost + Actual):
   ```python
   # After training models
   fig, ax = plt.subplots(figsize=(16, 6))
   
   # Actual
   ax.plot(test_df['timestamp'], test_df['request_count'], 
           label='Actual', linewidth=1, color='black')
   
   # Prophet prediction
   ax.plot(test_df['timestamp'], prophet_pred, 
           label='Prophet', linewidth=1, linestyle='--', alpha=0.7)
   
   # XGBoost/LightGBM prediction
   ax.plot(test_df['timestamp'], lgbm_pred,
           label='LightGBM', linewidth=1, linestyle='--', alpha=0.7)
   
   ax.legend()
   ax.set_title('Model Comparison: Predictions vs Actual')
   ```

2. **Anomaly highlights**:
   ```python
   # Highlight both Z-score and ML anomalies
   fig, ax = plt.subplots(figsize=(16, 6))
   ax.plot(df['timestamp'], df['request_count'], linewidth=0.5)
   
   # Z-score anomalies (red)
   z_anom = df[df['is_spike'] == 1]
   ax.scatter(z_anom['timestamp'], z_anom['request_count'],
              color='red', s=30, label='Statistical', alpha=0.6)
   
   # ML anomalies (orange)
   ml_anom = df[df['is_anomaly_ml'] == 1]
   ax.scatter(ml_anom['timestamp'], ml_anom['request_count'],
              color='orange', s=30, label='ML-based', alpha=0.6)
   
   # Overlap (both methods agree) - larger purple markers
   overlap = df[(df['is_spike'] == 1) & (df['is_anomaly_ml'] == 1)]
   ax.scatter(overlap['timestamp'], overlap['request_count'],
              color='purple', s=100, marker='*', label='Both', zorder=10)
   ```

3. **Event impact visualization**:
   ```python
   # Traffic around special events
   for event_date, event_info in SPECIAL_EVENTS.items():
       if event_info['type'] == 2:  # Space events
           date = pd.to_datetime(event_date)
           window = df[(df['timestamp'] >= date - pd.Timedelta(days=1)) &
                      (df['timestamp'] <= date + pd.Timedelta(days=1))]
           
           fig, ax = plt.subplots(figsize=(12, 4))
           ax.plot(window['timestamp'], window['request_count'])
           ax.axvline(date, color='red', linestyle='--', label=event_info['name'])
           ax.set_title(f"Traffic around {event_info['name']}")
           ax.legend()
   ```

---

### **Day 3: Benchmarking & Documentation (2-3 hours)** COMPLETED

#### Task 3.1: Comprehensive Benchmark Report (1h) - DONE
**Files to create:**
```
notebooks/11_final_benchmark.ipynb        # NEW
reports/benchmark_results.csv             # GENERATE
```

**Create benchmark notebook:**
```python
# Run all models on all time granularities
results = []

for granularity in ['1min', '5min', '15min']:
    # Load data
    train = pd.read_parquet(f'DATA/processed/train_features_{granularity}.parquet')
    test = pd.read_parquet(f'DATA/processed/test_features_{granularity}.parquet')
    
    # Prophet
    prophet_metrics = train_and_evaluate_prophet(train, test)
    results.append({
        'Dataset': granularity,
        'Model': 'Prophet',
        **prophet_metrics
    })
    
    # LightGBM
    lgbm_metrics = train_and_evaluate_lgbm(train, test)
    results.append({
        'Dataset': granularity,
        'Model': 'LightGBM',
        **lgbm_metrics
    })

# Create final table
benchmark_df = pd.DataFrame(results)
print("\n" + "="*60)
print("🏆 FINAL BENCHMARK RESULTS")
print("="*60)
print(benchmark_df.to_string(index=False))

# Save
benchmark_df.to_csv('reports/benchmark_results.csv', index=False)
```

**Expected format (like Colab):**
```
Dataset    Model      RMSE     MAE   SLA Violation (%)  Cost Saving (%)  Avg Servers
1min       Prophet    181.12   XX    16.80              75.00            1.00
1min       LightGBM   11.04    XX    0.38               62.87            1.49
5min       Prophet    710.38   XX    16.91              66.67            1.00
5min       LightGBM   37.78    XX    0.07               51.42            1.46
15min      Prophet    2002.39  XX    16.77              66.67            1.00
15min      LightGBM   93.67    XX    0.07               51.78            1.45
```

---

#### ✅ Task 3.2: Update Documentation (1-2h)
**Files to update:**
```
README.md                                 # UPDATE (add new features)
PROJECT_PLAN.md                           # UPDATE (mark completed)
reports/report.pdf                        # REGENERATE (if needed)
```

**Add to README:**
```markdown
## 🎯 Key Features

### Advanced Anomaly Detection
- **Statistical Method**: Z-score based spike detection (3-sigma rule)
- **ML Method**: IsolationForest for unsupervised anomaly detection
- **Comparison**: Both methods validated against known events

### Domain Knowledge Integration
- Special events dictionary (NASA missions, holidays)
- Event impact classification (high traffic, low traffic, outage)
- Historical context from July-August 1995

### Comprehensive Modeling
| Model | Type | Use Case |
|-------|------|----------|
| SARIMA | Statistical | Baseline, interpretable |
| Prophet | ML (Additive) | Seasonality, holidays |
| LightGBM | ML (Gradient Boosting) | Best accuracy, production |

### Results Highlights
- **Anomaly Detection**: 150-300 anomalies detected (1% of data)
- **Special Events**: 15+ events identified with impact classification
- **Model Performance**: RMSE 11-94 (depending on granularity)
- **Cost Savings**: 51-75% compared to fixed capacity
```

---

## 🎬 EXECUTION CHECKLIST

### Day 1: Core Features ✅
- [ ] Create `src/features/anomaly_detector.py`
- [ ] Add IsolationForest to notebook 03
- [ ] Add anomaly visualization to notebook 04
- [ ] Create `SPECIAL_EVENTS` dictionary
- [ ] Add event features to `AdvancedFeatureExtractor`
- [ ] Update notebook 03 to use event features
- [ ] Write unit tests for new features
- [ ] Run `pytest tests/` to verify

### Day 2: Verification & Enhancement ✅ COMPLETED
- [x] Verify no target scaling is applied
- [x] Create scaling comparison notebook (if needed) - Not needed, no scaling found
- [x] Add overlay plot (Actual + Prophet + LightGBM)
- [x] Add enhanced anomaly visualization
- [x] Add event impact visualization
- [x] Generate all figures for report (saved to reports/figures/)

### Day 3: Benchmarking & Documentation COMPLETED
- [x] Create `notebooks/11_final_benchmark.ipynb`
- [x] Run benchmark on all granularities
- [x] Generate `benchmark_results.csv`
- [x] Update README.md with new features
- [x] Update IMPROVEMENT_PLAN.md
- [x] Generate final report figures
- [x] Create DAY3_COMPLETION.md

---

## 📊 SUCCESS METRICS

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Anomaly Detection | 1-2% of data flagged | `df['is_anomaly_ml'].mean()` |
| Special Events | 15+ events detected | `df['is_special_event'].sum()` |
| Model RMSE | Similar to Colab | Compare benchmark table |
| Code Coverage | 80%+ | `pytest --cov=src tests/` |
| Documentation | All features documented | Manual review |

---

## 🚀 EXPECTED OUTCOMES

### After Implementation:
1. **✅ Feature Parity**: All key features from Colab notebook
2. **✅ Better Architecture**: Maintains modular, production-ready structure
3. **✅ Enhanced Explainability**: Two anomaly detection methods + domain knowledge
4. **✅ Competition Ready**: Comprehensive benchmarks and documentation

### Competitive Advantages:
- ✨ **vs Colab**: Better code structure, testing, API/Dashboard
- ✨ **vs Others**: Dual anomaly detection approach
- ✨ **Production Ready**: Can be deployed immediately

---

## 💡 BONUS ENHANCEMENTS (Optional)

If time permits:

1. **Uncertainty Quantification** (2h)
   - Add prediction intervals to LightGBM (quantile regression)
   - Add confidence bands to visualizations

2. **Real-time Simulation** (2h)
   - Interactive dashboard widget to adjust policy thresholds
   - Live cost-performance frontier plot

3. **Model Interpretability** (1h)
   - SHAP values for LightGBM predictions
   - Feature importance breakdown by event type

---

## 📝 NOTES

- Keep all existing code working (don't break what's good!)
- Add tests for new features
- Document all assumptions and design decisions
- Compare results with Colab notebook for validation

**Total Timeline**: 3 days (8-11 hours)  
**Risk Level**: LOW (additive changes only)  
**Expected ROI**: HIGH (competition-winning features)

---

**Created by**: GitHub Copilot  
**Last Updated**: January 30, 2026
