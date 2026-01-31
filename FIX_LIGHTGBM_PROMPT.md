# 🔧 PROMPT ĐỂ FIX LIGHTGBM OVERFITTING

**Ngày**: 31 Tháng 1, 2026  
**Mục đích**: Sửa LightGBM overfitting trong dự án autoscaling analysis

---

## 📋 PROMPT CHO CLAUDE CODE

Bạn hãy copy đoạn text bên dưới và paste vào Claude Code:

---

### **PROMPT BẮT ĐẦU TỪ ĐÂY** ⬇️

I need your help to fix a severe overfitting problem in my LightGBM model. Here's the context:

## 🎯 PROBLEM STATEMENT

**Current Model Performance (5min aggregation):**
- Validation RMSE: **0.53** (almost perfect)
- Test RMSE: **262.65** (worse than baseline!)
- Overfitting Ratio: **495x** (test/val)
- Test R²: **-3.59** (terrible)

**Baseline comparison:**
- Prophet Test RMSE: 139.19 ✅ (better)
- SARIMA Test RMSE: 150.37 ✅ (better)
- LightGBM Test RMSE: 262.65 ❌ (WORST!)

## 🔍 ROOT CAUSE ANALYSIS

I've identified 3 main issues:

### 1. **Data Leakage**
- Feature `request_count_pct_of_max` has correlation r=1.0 with target
- This feature is calculated FROM the target variable
- Model can "cheat" by using this feature

### 2. **Weak Regularization**
- Current Optuna found: `reg_lambda=0.000405` (almost zero!)
- Current Optuna found: `reg_alpha=0.27753` (too low)
- Model is too complex: `num_leaves=201` (very high)

### 3. **Bad Optuna Search Space**
- Using log scale for regularization allows extremely small values
- Current range: `(1e-8, 10.0, log=True)` → Optuna picks 0.0004
- Should use larger minimum values

## ✅ WHAT YOU NEED TO DO

Please help me fix the LightGBM model in this project by completing these 3 phases:

---

### **PHASE 1: Remove Data Leakage (Priority: HIGH)**

**File**: `notebooks/07_ml_models.ipynb`

**Task 1.1**: Find the cell where feature columns are selected (around line 200-250)

Look for code like:
```python
feature_cols = [col for col in df.columns if col not in ['timestamp', 'request_count', ...]]
```

**Replace with**:
```python
# Features to exclude (target + derived from target + metadata)
exclude_cols = [
    'timestamp',
    'request_count',           # target
    'request_count_pct_of_max' # DATA LEAKAGE - r=1.0 with target!
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"✅ Features used: {len(feature_cols)}")
print(f"✅ Removed 'request_count_pct_of_max' to prevent data leakage")
```

---

### **PHASE 2: Fix Optuna Search Space (Priority: HIGH)**

**File**: `notebooks/07_ml_models.ipynb`

**Task 2.1**: Find the Optuna objective function (around line 300-400)

Look for:
```python
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', ...),
        'reg_alpha': trial.suggest_float('reg_alpha', ...),
        'reg_lambda': trial.suggest_float('reg_lambda', ...),
        ...
    }
```

**Replace the regularization parameters with**:
```python
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # REDUCED max
        'max_depth': trial.suggest_int('max_depth', 3, 10),       # REDUCED max
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        
        # 🔥 KEY CHANGES: STRONGER REGULARIZATION
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 100.0),      # INCREASED min
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 100.0),    # INCREASED min
        
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'n_estimators': 100,  # Fixed for speed
        'random_state': 42,
        'verbose': -1
    }
    
    # Train model
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    return val_rmse
```

**Task 2.2**: Update Optuna study configuration

Find:
```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

**Replace with**:
```python
# Create study with better configuration
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42)
)

print("🔍 Starting Optuna hyperparameter tuning...")
print("   Trying 100 combinations (increased from 50)")
print("   Focus: Stronger regularization to prevent overfitting\n")

study.optimize(objective, n_trials=100, show_progress_bar=True)  # INCREASED trials

print("\n✅ Optuna tuning complete!")
print(f"   Best validation RMSE: {study.best_value:.2f}")
print(f"   Best parameters:")
for key, value in study.best_params.items():
    print(f"      {key}: {value}")
```

---

### **PHASE 3: Add Cross-Validation (Priority: MEDIUM)**

**File**: `notebooks/07_ml_models.ipynb`

**Task 3.1**: Add a new cell AFTER Optuna tuning to validate with CV

**Insert this code**:
```python
# ============================================================
# CROSS-VALIDATION CHECK (Prevent overfitting)
# ============================================================
print("=" * 60)
print("CROSS-VALIDATION CHECK")
print("=" * 60)

from sklearn.model_selection import TimeSeriesSplit

# Use best params from Optuna
best_params = study.best_params.copy()
best_params['n_estimators'] = 100
best_params['random_state'] = 42
best_params['verbose'] = -1

# Time series cross-validation (5 folds)
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

print("\nRunning 5-fold Time Series Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
    X_cv_train = X_train_scaled[train_idx]
    y_cv_train = y_train.iloc[train_idx]
    X_cv_val = X_train_scaled[val_idx]
    y_cv_val = y_train.iloc[val_idx]
    
    # Train model
    model_cv = lgb.LGBMRegressor(**best_params)
    model_cv.fit(X_cv_train, y_cv_train, 
                 eval_set=[(X_cv_val, y_cv_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    # Evaluate
    y_cv_pred = model_cv.predict(X_cv_val)
    fold_rmse = np.sqrt(mean_squared_error(y_cv_val, y_cv_pred))
    cv_scores.append(fold_rmse)
    print(f"   Fold {fold}: RMSE = {fold_rmse:.2f}")

print(f"\n📊 Cross-Validation Results:")
print(f"   Mean RMSE: {np.mean(cv_scores):.2f}")
print(f"   Std RMSE:  {np.std(cv_scores):.2f}")
print(f"   95% CI:    [{np.mean(cv_scores) - 2*np.std(cv_scores):.2f}, "
      f"{np.mean(cv_scores) + 2*np.std(cv_scores):.2f}]")
```

---

### **PHASE 4: Re-train Final Model (Priority: HIGH)**

**Task 4.1**: Re-train the final model with best params

Find the cell that trains the final model, and make sure it:
1. Uses the best params from Optuna
2. Includes early stopping
3. Saves the model

**Expected code**:
```python
# ============================================================
# FINAL MODEL TRAINING
# ============================================================
print("=" * 60)
print("TRAINING FINAL MODEL")
print("=" * 60)

# Train final model with best parameters
best_params = study.best_params.copy()
best_params['n_estimators'] = 500  # More iterations with early stopping
best_params['random_state'] = 42
best_params['verbose'] = -1

print("\n📊 Training with parameters:")
for key, value in best_params.items():
    print(f"   {key}: {value}")

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    callbacks=[
        lgb.early_stopping(100),  # Stop if no improvement for 100 rounds
        lgb.log_evaluation(50)     # Print every 50 rounds
    ]
)

print(f"\n✅ Training complete!")
print(f"   Best iteration: {final_model.best_iteration_}")
print(f"   Early stopped at: {final_model.n_estimators}")
```

---

### **PHASE 5: Comprehensive Evaluation (Priority: HIGH)**

**Task 5.1**: Add comprehensive evaluation on train/val/test sets

**Insert this code**:
```python
# ============================================================
# COMPREHENSIVE EVALUATION
# ============================================================
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

from src.utils.metrics import evaluate_forecast

# Evaluate on all sets
y_train_pred = final_model.predict(X_train_scaled)
y_val_pred = final_model.predict(X_val_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# Calculate metrics
train_metrics = evaluate_forecast(y_train, y_train_pred, name='Train')
val_metrics = evaluate_forecast(y_val, y_val_pred, name='Val')
test_metrics = evaluate_forecast(y_test, y_test_pred, name='Test')

# Print results
print("\n📊 PERFORMANCE SUMMARY:")
print("-" * 60)
print(f"{'Dataset':<10} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-" * 60)
print(f"{'Train':<10} {train_metrics['rmse']:<12.2f} {train_metrics['mae']:<12.2f} {train_metrics['r2']:<12.4f}")
print(f"{'Val':<10} {val_metrics['rmse']:<12.2f} {val_metrics['mae']:<12.2f} {val_metrics['r2']:<12.4f}")
print(f"{'Test':<10} {test_metrics['rmse']:<12.2f} {test_metrics['mae']:<12.2f} {test_metrics['r2']:<12.4f}")
print("-" * 60)

# Check for overfitting
overfit_ratio = test_metrics['rmse'] / val_metrics['rmse']
print(f"\n🔍 OVERFITTING CHECK:")
print(f"   Test/Val RMSE Ratio: {overfit_ratio:.2f}")
if overfit_ratio < 1.2:
    print(f"   ✅ GOOD - Model generalizes well!")
elif overfit_ratio < 2.0:
    print(f"   ⚠️ MODERATE - Some overfitting, but acceptable")
else:
    print(f"   ❌ BAD - Severe overfitting! Need more regularization")

# Compare with baselines
print(f"\n📈 COMPARISON WITH BASELINES:")
print(f"   Prophet Test RMSE: 139.19")
print(f"   SARIMA Test RMSE:  150.37")
print(f"   LightGBM Test RMSE: {test_metrics['rmse']:.2f}")
if test_metrics['rmse'] < 139.19:
    print(f"   ✅ LightGBM is BEST MODEL!")
elif test_metrics['rmse'] < 150.37:
    print(f"   ✅ LightGBM beats SARIMA!")
else:
    print(f"   ⚠️ LightGBM is not better than baselines")
```

---

## 🎯 EXPECTED RESULTS AFTER FIX

After implementing these fixes, we expect:

### **Before (Current - BAD)**
```
Model      Val RMSE  Test RMSE  Test/Val Ratio  Status
LightGBM      0.53     262.65         495x      ❌ Severe Overfit
```

### **After (Expected - GOOD)**
```
Model      Val RMSE  Test RMSE  Test/Val Ratio  Status
LightGBM     120-140   130-150       ~1.1x      ✅ Good Generalization
```

**Success criteria:**
- ✅ Test RMSE < 150 (better than SARIMA)
- ✅ Test RMSE < 140 (better than Prophet) 
- ✅ Test/Val ratio < 1.5 (good generalization)
- ✅ No data leakage warnings
- ✅ R² > 0.5 on test set

---

## 📋 VALIDATION CHECKLIST

After making the changes, please verify:

- [ ] Feature `request_count_pct_of_max` is removed from feature_cols
- [ ] Optuna uses `reg_alpha` and `reg_lambda` >= 1.0
- [ ] `num_leaves` <= 100
- [ ] Cross-validation is added and runs successfully
- [ ] Final model evaluation shows all 3 sets (train/val/test)
- [ ] Test/Val RMSE ratio is < 2.0
- [ ] Test RMSE is < 150 (better than SARIMA)
- [ ] Model is saved to `models/lgbm_tuned.pkl`
- [ ] Results are saved to `models/all_model_results.json`

---

## 🚀 HOW TO RUN AFTER FIX

After you make the changes, I'll run:

```bash
# Navigate to project
cd c:\Users\Admin\OneDrive\Documents\python\datafollow

# Open the notebook
jupyter notebook notebooks/07_ml_models.ipynb

# Or use VSCode
code notebooks/07_ml_models.ipynb

# Run all cells and check:
# 1. Data leakage warning should be gone
# 2. Optuna should find better params
# 3. Test RMSE should be < 150
```

---

## 📚 ADDITIONAL CONTEXT

**Project structure:**
```
datafollow/
├── notebooks/
│   ├── 07_ml_models.ipynb     # ← MAIN FILE TO EDIT
│   ├── 11_final_benchmark.ipynb
├── src/
│   ├── models/
│   │   ├── lgbm_model.py
│   ├── utils/
│   │   ├── metrics.py
├── models/
│   ├── all_model_results.json # ← Results will be saved here
```

**Key files involved:**
- `notebooks/07_ml_models.ipynb` - Main file to edit (Phases 1-5)
- `src/utils/metrics.py` - Already has `evaluate_forecast()` function
- `models/all_model_results.json` - Will be updated with new results

---

## ❓ QUESTIONS FOR YOU

Before you start, please confirm:
1. Are you comfortable editing Jupyter notebooks?
2. Should I also add XGBoost model like the Colab notebook?
3. Do you want me to create a comparison chart after the fix?

---

**Please implement all 5 phases and let me know the results. Thank you!**

### **PROMPT KẾT THÚC TẠI ĐÂY** ⬆️

---

## 📝 NOTES

- Bạn có thể copy toàn bộ phần giữa "PROMPT BẮT ĐẦU" và "PROMPT KẾT THÚC"
- Paste vào Claude Code hoặc bất kỳ AI assistant nào
- Prompt này có cấu trúc rõ ràng với:
  - Context đầy đủ
  - Root cause analysis
  - 5 phases chi tiết với code cụ thể
  - Expected results
  - Validation checklist

- Sau khi AI fix xong, bạn chạy lại notebook và kiểm tra kết quả
- Kỳ vọng Test RMSE giảm từ 262 xuống còn ~130-150

---

**Created by**: GitHub Copilot  
**Date**: January 31, 2026
