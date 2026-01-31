# KE HOACH NANG CAP - AUTOSCALING ANALYSIS

**Date**: January 31, 2026
**Version**: 2.0 (Updated based on codebase analysis)

---

## PHAN TICH HIEN TRANG THUC TE

### Ket qua khao sat codebase

| Component | File | Status | Chi tiet |
|-----------|------|--------|----------|
| Data Pipeline | notebooks/01-03 | OK | 15,264 train / 2,592 test rows |
| Feature Engineering | src/features/ | OK | 87 features, no leakage in lags |
| Prophet | notebooks/06 | OK | Test RMSE = 139.19 |
| SARIMA | notebooks/06 | OK | Test RMSE = 150.37 |
| **LightGBM** | notebooks/07 | **BROKEN** | Test RMSE = 262.65, R2 = -3.59 |
| Scaling Policies | src/scaling/ | OK | 3 policies implemented |

### Root Cause Analysis - LightGBM

**Van de chinh:**

1. **KHONG CO SCALER** - StandardScaler imported nhung KHONG DUOC SU DUNG
   - File: `notebooks/07_ml_models.ipynb` cell 1 co `from sklearn.preprocessing import StandardScaler`
   - Nhung KHONG co cell nao goi `scaler.fit_transform()`

2. **DATA LEAKAGE** - Feature `request_count_pct_of_max` co r=1.0 voi target
   - Day la feature duoc tinh tu target -> model "nhin thay" answer

3. **REGULARIZATION QUA YEU**
   - Optuna tim duoc: `reg_lambda=0.000405` (gan bang 0!)
   - `num_leaves=201` (qua cao, cho phep memorize data)

4. **OPTUNA SEARCH SPACE SAI**
   - Dung log scale cho regularization -> cho phep gia tri cuc nho
   - Range: `(1e-8, 10.0, log=True)` -> Optuna chon 0.0004

**Bang chung overfitting:**
```
Val RMSE:  0.53      (gan nhu perfect fit)
Test RMSE: 262.65    (te hon baseline!)
Ratio:     495x      (overfit nghiem trong)
```

---

## KE HOACH CHI TIET

### PHASE 1: FIX DATA LEAKAGE (30 phut)

#### Task 1.1: Loai bo feature `request_count_pct_of_max`

**File**: `notebooks/07_ml_models.ipynb`

Tim cell co code tuong tu:
```python
feature_cols = [col for col in df.columns if col not in ['timestamp', 'request_count', ...]]
```

**Sua thanh:**
```python
# Features to exclude (target + derived from target + metadata)
exclude_cols = [
    'timestamp',
    'request_count',           # target
    'request_count_pct_of_max' # DATA LEAKAGE - r=1.0 with target!
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Features used: {len(feature_cols)}")
print(f"Removed 'request_count_pct_of_max' to prevent data leakage")
```

---

### PHASE 2: THEM DATA SCALING (45 phut)

#### Task 2.1: Them cell scaling SAU khi split data

**Vi tri**: Sau cell train/val/test split, TRUOC cell training

```python
# ============================================================
# DATA SCALING - RobustScaler
# ============================================================
from sklearn.preprocessing import RobustScaler
import joblib

print("=" * 60)
print("DATA SCALING")
print("=" * 60)

# Khoi tao scaler
scaler = RobustScaler()

# QUAN TRONG: Chi fit tren TRAIN data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert lai thanh DataFrame (giu column names)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

# Luu scaler de dung khi predict
scaler_path = '../models/feature_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# Verify
print(f"\nTrain scaled - mean: {X_train_scaled.mean().mean():.4f}, std: {X_train_scaled.std().mean():.4f}")
print(f"Val scaled   - mean: {X_val_scaled.mean().mean():.4f}, std: {X_val_scaled.std().mean():.4f}")
print(f"Test scaled  - mean: {X_test_scaled.mean().mean():.4f}, std: {X_test_scaled.std().mean():.4f}")
```

#### Task 2.2: Cap nhat code training de dung scaled data

Tim cell training LightGBM va thay:
- `X_train` -> `X_train_scaled`
- `X_val` -> `X_val_scaled`
- `X_test` -> `X_test_scaled`

---

### PHASE 3: FIX LIGHTGBM CONFIG (1 gio)

#### Task 3.1: Thay doi baseline config

**Tim cell co `LGBMConfig` hoac `lgbm_params`**

**Config hien tai (SAI):**
```python
lgbm_config = LGBMConfig(
    num_leaves=31,
    max_depth=-1,           # Khong gioi han!
    learning_rate=0.05,
    reg_alpha=0.1,          # Qua yeu
    reg_lambda=0.1,         # Qua yeu
    min_child_samples=20,
    ...
)
```

**Config moi (DUNG):**
```python
lgbm_config = LGBMConfig(
    # Model complexity - GIAM MANH
    num_leaves=20,              # 31 -> 20
    max_depth=6,                # -1 -> 6 (gioi han)
    min_child_samples=50,       # 20 -> 50

    # Learning
    learning_rate=0.02,         # 0.05 -> 0.02 (cham hon)
    n_estimators=1500,          # Tang len vi learning rate giam

    # Regularization - TANG MANH
    reg_alpha=0.5,              # 0.1 -> 0.5 (L1)
    reg_lambda=2.0,             # 0.1 -> 2.0 (L2)

    # Subsampling - GIAM
    subsample=0.7,              # 0.8 -> 0.7
    subsample_freq=1,           # Enable subsampling moi iteration
    colsample_bytree=0.7,       # 0.8 -> 0.7

    # Other
    random_state=42,
    early_stopping_rounds=100,  # 50 -> 100
    verbose=-1,
)
```

#### Task 3.2: Fix Optuna search space

**Tim cell Optuna tuning**

**Search space hien tai (SAI):**
```python
'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
```

**Search space moi (DUNG):**
```python
def objective(trial):
    params = {
        # Complexity - gioi han chat
        'num_leaves': trial.suggest_int('num_leaves', 10, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 100),

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),

        # Regularization - LINEAR scale, gia tri LON
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),      # Khong dung log!
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),    # Khong dung log!

        # Subsampling
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
    }
    # ... rest of objective function
```

---

### PHASE 4: THEM OVERFITTING CHECK (30 phut)

#### Task 4.1: Them cell kiem tra overfitting

**Them SAU cell training:**
```python
# ============================================================
# OVERFITTING ANALYSIS
# ============================================================
print("=" * 60)
print("OVERFITTING CHECK")
print("=" * 60)

# Tinh metrics
train_pred = model.predict(X_train_scaled)
val_pred = model.predict(X_val_scaled)
test_pred = model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"\nRMSE Comparison:")
print(f"  Train: {train_rmse:.2f}")
print(f"  Val:   {val_rmse:.2f}")
print(f"  Test:  {test_rmse:.2f}")

print(f"\nOverfit Ratios:")
val_train_ratio = val_rmse / train_rmse
test_train_ratio = test_rmse / train_rmse
test_val_ratio = test_rmse / val_rmse

print(f"  Val/Train:  {val_train_ratio:.2f}x")
print(f"  Test/Train: {test_train_ratio:.2f}x")
print(f"  Test/Val:   {test_val_ratio:.2f}x")

# Verdict
if test_val_ratio < 1.5:
    print("\n[OK] Good generalization!")
elif test_val_ratio < 2.5:
    print("\n[WARN] Moderate overfitting - may need more regularization")
else:
    print("\n[ERROR] Severe overfitting - model is memorizing training data")
    print("  -> Increase reg_lambda and reg_alpha")
    print("  -> Reduce num_leaves and max_depth")
    print("  -> Check for data leakage")
```

#### Task 4.2: Them feature importance visualization

```python
# ============================================================
# FEATURE IMPORTANCE
# ============================================================
import matplotlib.pyplot as plt

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Top 20 features
top_n = 20
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(importance['feature'].head(top_n)[::-1],
        importance['importance'].head(top_n)[::-1])
ax.set_xlabel('Importance')
ax.set_title(f'Top {top_n} Feature Importance')
plt.tight_layout()
plt.savefig('../reports/figures/lgbm_feature_importance.png', dpi=150)
plt.show()

# Check for suspicious features
print("\nFeature Importance Analysis:")
print(f"  Total features: {len(feature_cols)}")
print(f"  Features with importance > 0: {(importance['importance'] > 0).sum()}")
print(f"  Features with importance = 0: {(importance['importance'] == 0).sum()}")

# Warning if one feature dominates
if importance['importance'].iloc[0] > 0.5:
    print(f"\n[WARN] Feature '{importance['feature'].iloc[0]}' dominates ({importance['importance'].iloc[0]:.2%})")
    print("  -> May indicate data leakage")
```

---

### PHASE 5: VERIFY VA BENCHMARK (1 gio)

#### Task 5.1: Chay lai notebook tu dau

1. Kernel -> Restart and Run All
2. Kiem tra ket qua moi

**Ket qua mong doi:**
```
LightGBM sau khi fix:
  Train RMSE: ~30-50
  Val RMSE:   ~80-120
  Test RMSE:  ~100-150  (target: < 150)
  Test R2:    > 0.3

  Overfit ratio (Test/Val): < 2x
```

#### Task 5.2: So sanh 3 models

**Tao/update cell cuoi notebook:**
```python
# ============================================================
# FINAL MODEL COMPARISON
# ============================================================
print("=" * 60)
print("MODEL BENCHMARK - TEST SET")
print("=" * 60)

results = {
    'Prophet': {'RMSE': 139.19, 'MAE': 102.52, 'R2': -0.29},
    'SARIMA':  {'RMSE': 150.37, 'MAE': 108.56, 'R2': -0.50},
    'LightGBM': {'RMSE': test_rmse, 'MAE': test_mae, 'R2': test_r2},
}

results_df = pd.DataFrame(results).T
print(results_df.round(2))

# Best model
best_model = results_df['RMSE'].idxmin()
print(f"\nBest Model: {best_model} (RMSE: {results_df.loc[best_model, 'RMSE']:.2f})")

# Visualization
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(results_df.index, results_df['RMSE'], color=colors)
ax.set_ylabel('RMSE (lower is better)')
ax.set_title('Model Comparison - Test Set')

for bar, rmse in zip(bars, results_df['RMSE']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{rmse:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/figures/model_comparison.png', dpi=150)
plt.show()
```

---

## CHECKLIST THUC HIEN

### Phase 1: Data Leakage
- [ ] Loai bo `request_count_pct_of_max` khoi feature_cols
- [ ] Verify so luong features giam tu 87 -> 86

### Phase 2: Data Scaling
- [ ] Them cell RobustScaler
- [ ] Verify scaler chi fit tren train
- [ ] Luu scaler vao file
- [ ] Cap nhat code dung X_train_scaled, X_val_scaled, X_test_scaled

### Phase 3: LightGBM Config
- [ ] Cap nhat baseline config (num_leaves=20, max_depth=6, reg_lambda=2.0)
- [ ] Cap nhat Optuna search space (linear scale cho regularization)

### Phase 4: Verification
- [ ] Them overfitting check cell
- [ ] Them feature importance visualization
- [ ] Run notebook tu dau

### Phase 5: Benchmark
- [ ] Test RMSE < 150
- [ ] Test R2 > 0.3
- [ ] Overfit ratio < 2x
- [ ] Tao comparison chart

---

## KET QUA MONG DOI

| Metric | Truoc | Sau | Thay doi |
|--------|-------|-----|----------|
| Test RMSE | 262.65 | < 150 | -43% |
| Test R2 | -3.59 | > 0.3 | +3.89 |
| Val/Test ratio | 495x | < 2x | Giam 247x |
| Features | 87 | 86 | -1 (leakage) |

---

## TROUBLESHOOTING

### Neu Test RMSE van > 150:

1. **Tang regularization hon nua:**
   - `reg_lambda=5.0` (thay vi 2.0)
   - `reg_alpha=1.0` (thay vi 0.5)

2. **Giam complexity hon nua:**
   - `num_leaves=15` (thay vi 20)
   - `max_depth=5` (thay vi 6)
   - `min_child_samples=100` (thay vi 50)

3. **Kiem tra them data leakage:**
   - Chay correlation analysis voi target
   - Loai bo features co r > 0.95

4. **Thu StandardScaler thay vi RobustScaler:**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   ```

### Neu RMSE tot nhung R2 van am:

- R2 am nghia la model te hon baseline (mean prediction)
- Voi time series co nhieu noise, R2 am la binh thuong
- Focus vao RMSE va MAE de danh gia

---

## FILES CAN CHINH SUA

| File | Thay doi |
|------|----------|
| `notebooks/07_ml_models.ipynb` | Chinh sua chinh |
| `models/feature_scaler.pkl` | Tao moi (luu scaler) |
| `reports/figures/lgbm_feature_importance.png` | Tao moi |
| `reports/figures/model_comparison.png` | Tao moi |

---

## GHI CHU QUAN TRONG

1. **KHONG ghi de X_train goc** - Tao bien moi X_train_scaled
2. **Luu scaler** - Can cho production prediction
3. **Optuna linear scale** - Khong dung log scale cho regularization
4. **Check feature importance** - Neu 1 feature dominate > 50% -> co the data leakage

---

Last updated: 2026-01-31 (v2.0 - based on actual codebase analysis)
