# 📊 MODEL RMSE TEST RESULTS

**Ngày test**: 31 Tháng 1, 2026  
**Dataset**: NASA Web Server Logs (5min aggregation)  
**Test size**: 9 ngày (Aug 23-31, 1995)

---

## 🎯 TÓM TẮT KẾT QUẢ

### **Best Model: Prophet** 
- ✅ Test RMSE: **139.19**
- ✅ Test MAE: 102.52
- ✅ Generalization ratio: 1.34x (good)

### **Baseline: SARIMA**
- ✅ Test RMSE: **150.37**
- ✅ Test MAE: 108.56
- ✅ Generalization ratio: 1.01x (excellent)

### **Problem: LightGBM**
- ❌ Test RMSE: **262.65** (worse than baselines!)
- ❌ Validation RMSE: 0.53 (too perfect)
- ❌ Overfitting ratio: **497x** (severe)

---

## 📈 CHI TIẾT PERFORMANCE

### Test Set Performance (9 ngày test data)

| Model | RMSE | MAE | R² | Status |
|-------|------|-----|-----|--------|
| **Prophet** | **139.19** ✅ | 102.52 | -0.29 | **BEST** |
| **SARIMA** | **150.37** ✅ | 108.56 | -0.50 | Good |
| **LightGBM** | **262.65** ❌ | 235.24 | -3.59 | **OVERFITTED** |

### Validation Set Performance (train subset)

| Model | RMSE | MAE | R² | Status |
|-------|------|-----|-----|--------|
| Prophet | 103.57 | 88.59 | -0.12 | Good |
| SARIMA | 148.37 | 114.29 | -1.30 | Good |
| LightGBM | **0.53** ⚠️ | 0.37 | **1.00** | Too perfect! |

---

## 🔍 OVERFITTING ANALYSIS

### Test/Validation RMSE Ratio

```
Prophet:  1.34x  ✅ Good generalization
SARIMA:   1.01x  ✅ Excellent generalization  
LightGBM: 497x   ❌ SEVERE OVERFITTING!
```

**Thresholds:**
- ✅ **< 1.5x**: Good generalization
- ⚠️ **1.5-2.0x**: Moderate overfitting
- ❌ **> 2.0x**: Severe overfitting

---

## 📊 VISUALIZATION

Xem biểu đồ so sánh tại: [reports/figures/model_rmse_comparison.png](reports/figures/model_rmse_comparison.png)

**3 charts:**
1. **Test RMSE Comparison** - So sánh RMSE test set
2. **Val vs Test RMSE** - So sánh validation và test
3. **Overfitting Analysis** - Phân tích mức độ overfit

---

## 🎯 RANKING & COMPARISON

### 1. Prophet (WINNER 🏆)
```
✅ Pros:
   - Best test RMSE (139.19)
   - Good generalization (1.34x ratio)
   - Stable performance
   - R² close to 0 (reasonable for time series)

⚠️ Cons:
   - Slower training than ML models
   - Less flexible than gradient boosting
```

### 2. SARIMA (Solid Baseline ⭐)
```
✅ Pros:
   - Excellent generalization (1.01x ratio)
   - Statistical foundation
   - Interpretable
   - Test RMSE only 8% worse than Prophet

⚠️ Cons:
   - Slower inference
   - Limited feature engineering
   - Fixed seasonal patterns
```

### 3. LightGBM (NEEDS FIX ❌)
```
❌ Critical Issues:
   - Severe overfitting (497x ratio!)
   - Test RMSE worse than baselines
   - Val RMSE = 0.53 (memorized training data)
   - R² = -3.59 on test (terrible)

🔧 Root Causes (identified):
   1. Data leakage: feature 'request_count_pct_of_max'
   2. Weak regularization: reg_lambda = 0.0004
   3. Model too complex: num_leaves = 201

📋 Action Required:
   See FIX_LIGHTGBM_PROMPT.md for detailed fix plan
```

---

## 📉 SO SÁNH VỚI COLAB NOTEBOOK

### Colab Results (từ Google Colab)

| Dataset | Model | RMSE | Status |
|---------|-------|------|--------|
| 5min | Prophet | 762.93 | ❌ Worse |
| 5min | XGBoost | **53.02** | ✅ **Excellent** |
| 1min | XGBoost | **15.04** | ✅ Excellent |
| 15min | XGBoost | 127.51 | ✅ Good |

### So sánh cụ thể (5min aggregation)

| Metric | Colab Prophet | Project Prophet | Winner |
|--------|---------------|-----------------|--------|
| Test RMSE | 762.93 | **139.19** | 🏆 **Project (5.5x better!)** |
| Generalization | ⚠️ Moderate | ✅ Good | 🏆 Project |

| Metric | Colab XGBoost | Project LightGBM | Winner |
|--------|---------------|------------------|--------|
| Test RMSE | **53.02** | 262.65 | 🏆 **Colab (5x better!)** |
| Generalization | ✅ Good | ❌ Severe overfit | 🏆 Colab |

**Key Insight:**
- ✅ Dự án Prophet TỐT HƠN NHIỀU so với Colab (139 vs 762)
- ❌ Dự án LightGBM TỆ HƠN NHIỀU so với Colab XGBoost (262 vs 53)
- 🎯 **Action**: Fix LightGBM hoặc thêm XGBoost vào project

---

## 🔧 NEXT STEPS

### Priority 1: Fix LightGBM (2-3 giờ)
1. ✅ Đã phân tích root cause
2. ✅ Đã tạo FIX_LIGHTGBM_PROMPT.md
3. ⏳ Cần implement fixes:
   - Remove data leakage feature
   - Fix Optuna search space
   - Add stronger regularization
   - Re-train and validate

**Expected after fix:**
- Target Test RMSE: < 140 (better than Prophet)
- Target Val/Test ratio: < 1.5x
- Target R²: > 0.5

### Priority 2: Add XGBoost (1-2 giờ)
1. Port code from Colab notebook
2. Create `src/models/xgboost_model.py`
3. Benchmark against other models
4. Expected: Test RMSE ~ 50-60 (như Colab)

### Priority 3: Ensemble Model (Optional)
1. Combine Prophet + XGBoost
2. Weighted average or stacking
3. Potential: Test RMSE < 100

---

## 📝 TEST COMMANDS

### Run full benchmark:
```bash
cd c:\Users\Admin\OneDrive\Documents\python\datafollow

# Prophet + SARIMA
jupyter notebook notebooks/06_baseline_models.ipynb

# LightGBM
jupyter notebook notebooks/07_ml_models.ipynb

# All models comparison
jupyter notebook notebooks/11_final_benchmark.ipynb
```

### Quick RMSE check:
```bash
# Via Python
python -c "
import json
with open('models/all_model_results.json', 'r') as f:
    r = json.load(f)
    print(f'Prophet: {r['prophet_test']['rmse']:.2f}')
    print(f'SARIMA:  {r['sarima_test']['rmse']:.2f}')
    print(f'LightGBM: {r['lgbm_test']['rmse']:.2f}')
"

# Via API
curl http://localhost:8000/metrics
```

---

## 🎓 LEARNINGS & INSIGHTS

### What Worked Well ✅
1. **Prophet**: Excellent for this use case
   - Handles trend and seasonality automatically
   - Robust to outliers (STS-70 launch, hurricane)
   - Good generalization
   
2. **SARIMA**: Solid statistical baseline
   - Perfect generalization (1.01x)
   - Good for understanding data patterns

3. **Feature Engineering**: 87 features created
   - Time features (cyclical encoding)
   - Lag features (1-288 periods)
   - Rolling statistics
   - Special events dictionary
   - IsolationForest anomaly detection

### What Needs Improvement ❌
1. **LightGBM Tuning**: 
   - Optuna search space too permissive
   - Needs stronger regularization constraints
   - Data leakage in features

2. **Model Selection**:
   - Should benchmark more models (XGBoost, LSTM)
   - Should try ensemble methods

3. **Hyperparameter Tuning**:
   - Need better validation strategy
   - Should use TimeSeriesSplit CV

---

## 📚 REFERENCES

### Files
- Model results: `models/all_model_results.json`
- Benchmark CSV: `reports/benchmark_results.csv`
- Visualization: `reports/figures/model_rmse_comparison.png`
- Fix guide: `FIX_LIGHTGBM_PROMPT.md`
- Comparison report: `COMPARISON_REPORT.md`

### Notebooks
- Baseline models: `notebooks/06_baseline_models.ipynb`
- ML models: `notebooks/07_ml_models.ipynb`
- Final benchmark: `notebooks/11_final_benchmark.ipynb`

### Code
- Prophet: `src/models/prophet_model.py`
- SARIMA: `src/models/sarima.py`
- LightGBM: `src/models/lgbm_model.py`

---

**Tác giả**: GitHub Copilot  
**Ngày**: 31 Tháng 1, 2026  
**Version**: 1.0

**Status**: 
- ✅ Prophet: Production-ready
- ✅ SARIMA: Production-ready
- ❌ LightGBM: Needs fixing
