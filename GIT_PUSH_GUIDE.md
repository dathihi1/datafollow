# Quick Start - Git Commands

## 🚀 Push lên GitHub

### Bước 1: Cấu hình Git (chỉ làm 1 lần)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Bước 2: Tạo repository trên GitHub
1. Vào https://github.com/new
2. Tên repo: `datafollow` hoặc `autoscaling-analysis`
3. Description: "Autoscaling Analysis for NASA Web Server Logs - DATAFLOW 2026"
4. Chọn: ✅ Public hoặc Private
5. Không check "Initialize with README" (vì đã có)
6. Click "Create repository"

### Bước 3: Add và Commit local
```bash
cd c:\Users\Admin\OneDrive\Documents\python\datafollow

# Add tất cả files
git add .

# Commit
git commit -m "Initial commit: Autoscaling Analysis project

- 3 ML models: Prophet (RMSE=139), SARIMA, LightGBM
- 87 features with domain knowledge
- FastAPI REST API (8 endpoints)
- Streamlit dashboard
- 184 unit tests (45% coverage)
- Docker deployment ready
- Complete documentation (10+ MD files)"

# Check status
git status
```

### Bước 4: Push lên GitHub
```bash
# Link remote (thay YOUR_USERNAME bằng GitHub username của bạn)
git remote add origin https://github.com/YOUR_USERNAME/datafollow.git

# Đổi branch sang main (GitHub default)
git branch -M main

# Push
git push -u origin main
```

### Bước 5: Verify
- Vào https://github.com/YOUR_USERNAME/datafollow
- Kiểm tra files đã upload
- ✅ Models `.pkl` KHÔNG có (theo .gitignore)
- ✅ Có file MODELS_DOWNLOAD.md hướng dẫn tải

---

## 📋 What's Included in Git

### ✅ Included (pushed to GitHub):
```
✅ Source code (src/, app/, tests/)
✅ Notebooks (11 notebooks)
✅ Documentation (10+ MD files)
✅ Configuration (pyproject.toml, requirements.txt)
✅ Docker files (Dockerfile, docker-compose.yml)
✅ Deployment scripts (digitalocean/)
✅ Model configs (models/*.json)
✅ .gitignore
✅ README.md, OVERVIEW.md, etc.
```

### ❌ Excluded (not pushed, in .gitignore):
```
❌ Models (models/*.pkl) - 2GB total!
❌ Data files (DATA/*.txt) - 359MB
❌ Processed data (DATA/processed/*.parquet)
❌ Notebook outputs (.ipynb_checkpoints)
❌ Python cache (__pycache__/)
❌ Virtual environment (.venv/)
❌ IDE files (.vscode/, .idea/)
```

---

## 📦 Upload Models (Separate)

Models are too large for GitHub. Upload to:

### Option 1: GitHub Releases (Recommended)
```bash
# 1. Create a release on GitHub
# Go to: https://github.com/YOUR_USERNAME/datafollow/releases/new
# Tag: v1.0.0
# Title: "v1.0.0 - Pre-trained Models"

# 2. Attach files:
#    - prophet_5m.pkl (1.6 MB)
#    - lgbm_5m.pkl (10.7 MB)
#    - feature_scaler.pkl (100 KB)
#    - sarima_5m.pkl (2 GB) - Optional, might hit limit

# 3. Users download via:
wget https://github.com/YOUR_USERNAME/datafollow/releases/download/v1.0.0/prophet_5m.pkl
```

### Option 2: Google Drive
```bash
# 1. Upload models/ folder to Google Drive
# 2. Get shareable link
# 3. Update MODELS_DOWNLOAD.md with link
# 4. Commit updated MODELS_DOWNLOAD.md
```

### Option 3: Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.pkl"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Now add models
git add models/*.pkl
git commit -m "Add pre-trained models"
git push
```

---

## 🔄 Update Code Later

```bash
# Make changes to code
# ... edit files ...

# Stage changes
git add .

# Commit
git commit -m "Fix: LightGBM overfitting issue"

# Push
git push origin main
```

---

## 📊 File Sizes Summary

```
Repository (without models):  ~50 MB
  ├─ Source code:              ~2 MB
  ├─ Notebooks:                ~15 MB
  ├─ Documentation:            ~1 MB
  ├─ Tests:                    ~500 KB
  └─ Dependencies:             ~30 MB (not in git)

Models (separate):            ~2 GB
  ├─ prophet_5m.pkl:           1.6 MB   ✅
  ├─ lgbm_5m.pkl:              10.7 MB  ✅
  ├─ sarima_5m.pkl:            2.0 GB   ⚠️
  └─ feature_scaler.pkl:       100 KB   ✅

Data (not in git):            ~400 MB
  ├─ train.txt:                ~300 MB
  ├─ test.txt:                 ~60 MB
  └─ processed/:               ~40 MB
```

---

## ✅ Final Checklist

Before pushing:
- [ ] Đã cấu hình git user.name và user.email
- [ ] Đã tạo repository trên GitHub
- [ ] Đã review git status (check files)
- [ ] Đã commit với message rõ ràng
- [ ] Đã add remote origin
- [ ] Đã push lên GitHub
- [ ] Verify files trên GitHub
- [ ] Models .pkl KHÔNG có trong repo ✅
- [ ] Update MODELS_DOWNLOAD.md với link download
- [ ] Test clone và setup từ GitHub

After pushing:
- [ ] Upload models to GitHub Releases / Google Drive
- [ ] Update README.md với link repository
- [ ] Add repository to competition submission
- [ ] Share with team/reviewers

---

**Ready to push? Run:**
```bash
cd c:\Users\Admin\OneDrive\Documents\python\datafollow
git add .
git commit -m "Initial commit: Autoscaling Analysis project"
git remote add origin https://github.com/YOUR_USERNAME/datafollow.git
git branch -M main
git push -u origin main
```

**Then go to:** https://github.com/YOUR_USERNAME/datafollow
