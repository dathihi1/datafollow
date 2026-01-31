# 📦 MODELS DOWNLOAD GUIDE

**Project**: Autoscaling Analysis  
**Date**: January 31, 2026

---

## ⚠️ IMPORTANT: Models Not Included in Git

Model files are **NOT included** in the git repository due to their large size:

| File | Size | Reason |
|------|------|--------|
| `sarima_5m.pkl` | **2.0 GB** | Too large for GitHub |
| `lgbm_5m.pkl` | 10.7 MB | Above recommended limit |
| `prophet_5m.pkl` | 1.6 MB | Included but optional |
| `feature_scaler.pkl` | < 1 MB | ✅ Included in git |

**Total**: ~2.01 GB of model files

---

## 🚀 QUICK START (3 Options)

### Option 1: Download Pre-trained Models (Recommended)

**From Google Drive / Dropbox / GitHub Releases:**

```bash
# 1. Navigate to project directory
cd datafollow/models/

# 2. Download from shared link (example)
# Replace with your actual download link
wget https://your-storage-url/models/prophet_5m.pkl
wget https://your-storage-url/models/sarima_5m.pkl
wget https://your-storage-url/models/lgbm_5m.pkl
wget https://your-storage-url/models/feature_scaler.pkl

# 3. Verify downloads
ls -lh *.pkl
```

**Expected file sizes:**
```
prophet_5m.pkl       ~1.6 MB   ✅
sarima_5m.pkl        ~2.0 GB   ⚠️ Large!
lgbm_5m.pkl          ~10.7 MB  ✅
feature_scaler.pkl   ~100 KB   ✅
```

---

### Option 2: Train Models Yourself (2-3 hours)

**Run the training notebooks:**

```bash
# 1. Make sure you have the data
# Download train.txt and test.txt first (see DATA_DOWNLOAD.md)

# 2. Open notebooks in order
jupyter notebook

# 3. Run these notebooks:
notebooks/01_data_ingestion.ipynb      # Parse logs (10 min)
notebooks/02_aggregation.ipynb         # Aggregate (5 min)
notebooks/03_feature_engineering.ipynb # Features (15 min)

# 4. Train models
notebooks/06_baseline_models.ipynb     # Prophet + SARIMA (1-2 hours)
notebooks/07_ml_models.ipynb           # LightGBM (30-60 min)

# Models will be saved to models/ automatically
```

**Hardware requirements for training:**
- **RAM**: 8 GB minimum, 16 GB recommended
- **CPU**: Multi-core recommended (SARIMA is slow)
- **Disk**: 5 GB free space
- **Time**: 2-3 hours total

---

### Option 3: Use Cloud Storage (Best for Teams)

**Setup with AWS S3 / Google Cloud Storage:**

```bash
# 1. Install cloud CLI
pip install awscli  # or google-cloud-storage

# 2. Configure credentials
aws configure

# 3. Download models
aws s3 cp s3://your-bucket/datafollow-models/ models/ --recursive

# 4. Verify
ls -lh models/*.pkl
```

---

## 📋 STEP-BY-STEP SETUP GUIDE

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/datafollow.git
cd datafollow
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 3: Create Models Directory Structure

```bash
# Create directories if not exists
mkdir -p models
mkdir -p DATA/processed
mkdir -p reports/figures

# Add .gitkeep files
touch models/.gitkeep
touch DATA/processed/.gitkeep
touch reports/figures/.gitkeep
```

### Step 4: Download Models

**Choose one method:**

**A. Manual Download (Google Drive example)**
1. Go to: https://drive.google.com/your-shared-folder
2. Download all `.pkl` files
3. Move to `datafollow/models/`

**B. Command Line (wget example)**
```bash
cd models/

# Prophet model (1.6 MB)
wget -O prophet_5m.pkl "https://your-storage-url/prophet_5m.pkl"

# SARIMA model (2 GB - this will take time!)
wget -O sarima_5m.pkl "https://your-storage-url/sarima_5m.pkl"

# LightGBM model (10.7 MB)
wget -O lgbm_5m.pkl "https://your-storage-url/lgbm_5m.pkl"

# Feature scaler (small)
wget -O feature_scaler.pkl "https://your-storage-url/feature_scaler.pkl"
```

**C. Python Script**
```python
# download_models.py
import requests
import os

MODELS = {
    'prophet_5m.pkl': 'https://your-storage-url/prophet_5m.pkl',
    'sarima_5m.pkl': 'https://your-storage-url/sarima_5m.pkl',
    'lgbm_5m.pkl': 'https://your-storage-url/lgbm_5m.pkl',
    'feature_scaler.pkl': 'https://your-storage-url/feature_scaler.pkl',
}

for filename, url in MODELS.items():
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    with open(f'models/{filename}', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ {filename} downloaded")

print("\n🎉 All models downloaded!")
```

### Step 5: Verify Installation

```bash
# Check files exist
ls -lh models/

# Expected output:
# prophet_5m.pkl        1.6M
# sarima_5m.pkl         2.0G
# lgbm_5m.pkl           10.7M
# feature_scaler.pkl    100K
# *.json                (config files)
```

### Step 6: Test Models

```python
# test_models.py
import pickle
import os

models_to_test = [
    'models/prophet_5m.pkl',
    'models/sarima_5m.pkl',
    'models/lgbm_5m.pkl',
    'models/feature_scaler.pkl',
]

print("🧪 Testing models...\n")

for model_path in models_to_test:
    if not os.path.exists(model_path):
        print(f"❌ Missing: {model_path}")
        continue
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ {model_path} - Loaded successfully")
    except Exception as e:
        print(f"❌ {model_path} - Error: {e}")

print("\n🎉 Model verification complete!")
```

---

## 🔍 TROUBLESHOOTING

### Issue 1: SARIMA Model Too Large

**Problem**: `sarima_5m.pkl` is 2 GB!

**Solution A**: Skip SARIMA, use Prophet + LightGBM only
```python
# SARIMA is just baseline for comparison
# Prophet (RMSE=139) is better anyway
# You can skip SARIMA if disk space is limited
```

**Solution B**: Use Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "models/*.pkl"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Now you can commit large files
git add models/sarima_5m.pkl
git commit -m "Add SARIMA model"
git push
```

**Solution C**: Use DVC (Data Version Control)
```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Add models to DVC
dvc add models/sarima_5m.pkl

# DVC will create models/sarima_5m.pkl.dvc (small file)
# Commit this instead
git add models/sarima_5m.pkl.dvc
git commit -m "Add SARIMA model with DVC"

# Setup remote storage
dvc remote add -d myremote s3://mybucket/path
dvc push
```

---

### Issue 2: Download Failed / Corrupted File

**Check file integrity:**
```bash
# Check file size
ls -lh models/sarima_5m.pkl

# If size is wrong, re-download
rm models/sarima_5m.pkl
wget -O models/sarima_5m.pkl "https://your-url/sarima_5m.pkl"
```

**Verify with MD5 checksum:**
```bash
# Generate checksum (once, after training)
md5sum models/sarima_5m.pkl > models/sarima_5m.pkl.md5

# Verify after download
md5sum -c models/sarima_5m.pkl.md5
```

---

### Issue 3: Out of Memory When Loading SARIMA

**Problem**: SARIMA model is too large for available RAM

**Solution**:
```python
# Don't load SARIMA if you don't need it
# Use Prophet instead (smaller, better performance)

# Or load only when needed
import gc
import pickle

def predict_with_sarima(data):
    # Load model
    with open('models/sarima_5m.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make prediction
    predictions = model.predict(data)
    
    # Free memory
    del model
    gc.collect()
    
    return predictions
```

---

## 📦 MODEL FILE DETAILS

### Prophet Model (`prophet_5m.pkl`)
```yaml
Size: 1.6 MB
Type: Facebook Prophet model
Training time: ~10 minutes
Inference time: ~0.1 sec per prediction
Best for: Stable, production use
Performance: RMSE = 139.19 (BEST)
```

### SARIMA Model (`sarima_5m.pkl`)
```yaml
Size: 2.0 GB ⚠️
Type: Statsmodels SARIMAX
Training time: ~1-2 hours (slow!)
Inference time: ~1 sec per prediction
Best for: Statistical baseline
Performance: RMSE = 150.37
Note: Large because stores full state
```

### LightGBM Model (`lgbm_5m.pkl`)
```yaml
Size: 10.7 MB
Type: LightGBM Regressor
Training time: ~30 minutes
Inference time: ~0.01 sec per prediction
Best for: Fast inference
Performance: RMSE = 262.65 (overfitted)
Note: Needs fixing (see FIX_LIGHTGBM_PROMPT.md)
```

### Feature Scaler (`feature_scaler.pkl`)
```yaml
Size: ~100 KB
Type: RobustScaler (sklearn)
Purpose: Scale features before LightGBM
Required by: LightGBM model only
```

---

## 🌐 HOSTING OPTIONS FOR LARGE FILES

### Option 1: Google Drive (Free, Easy)
1. Upload models to Google Drive
2. Get shareable link
3. Use `gdown` to download:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=YOUR_FILE_ID
```

### Option 2: GitHub Releases (Free, 2GB limit)
1. Create a release on GitHub
2. Attach model files to release
3. Download via release URL
```bash
wget https://github.com/user/repo/releases/download/v1.0/sarima_5m.pkl
```

### Option 3: AWS S3 (Paid, Professional)
```bash
# Upload
aws s3 cp models/ s3://mybucket/models/ --recursive

# Download
aws s3 cp s3://mybucket/models/ models/ --recursive
```

### Option 4: Dropbox (Free, 2GB)
```bash
# Get direct download link (replace dl=0 with dl=1)
wget -O models/sarima_5m.pkl "https://www.dropbox.com/s/xyz/sarima_5m.pkl?dl=1"
```

---

## ✅ FINAL CHECKLIST

Before running the application:

- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models directory exists
- [ ] `prophet_5m.pkl` downloaded (1.6 MB)
- [ ] `lgbm_5m.pkl` downloaded (10.7 MB)
- [ ] `feature_scaler.pkl` downloaded (100 KB)
- [ ] `sarima_5m.pkl` downloaded (2 GB) - Optional
- [ ] Models tested with `test_models.py`
- [ ] API starts successfully (`uvicorn src.api.main:app`)

---

## 📞 NEED HELP?

If you have issues downloading or loading models:

1. **Check file size**: Compare with expected sizes above
2. **Check free disk space**: Need ~5 GB total
3. **Check RAM**: Need 8 GB+ for SARIMA
4. **Skip SARIMA**: Use Prophet instead (better performance, smaller)
5. **Re-train models**: Follow notebooks (2-3 hours)
6. **Open an issue**: https://github.com/your-username/datafollow/issues

---

## 🚀 QUICK COMMANDS SUMMARY

```bash
# Setup project
git clone https://github.com/your-username/datafollow.git
cd datafollow
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download models (replace URLs with your own)
cd models
wget https://your-url/prophet_5m.pkl
wget https://your-url/lgbm_5m.pkl
wget https://your-url/feature_scaler.pkl
# wget https://your-url/sarima_5m.pkl  # Optional

# Test
cd ..
python test_models.py

# Run API
uvicorn src.api.main:app --reload

# ✅ Done!
```

---

**Last Updated**: January 31, 2026  
**Contact**: your.email@example.com
