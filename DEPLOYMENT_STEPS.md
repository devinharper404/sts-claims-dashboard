# 🚀 STS Claims Dashboard - Deployment Guide

## Quick Deployment to Streamlit Cloud

### Step 1: Create GitHub Repository
1. Go to [GitHub.com](https://github.com)
2. Click "New Repository"
3. Name: `sts-claims-dashboard`
4. Make it **Public**
5. Click "Create Repository"

### Step 2: Push Your Code
Replace `YOUR_GITHUB_USERNAME` with your actual username:

```bash
cd c:\Users\311741\sts
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/sts-claims-dashboard.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Fill in the form:

**Repository:** `YOUR_GITHUB_USERNAME/sts-claims-dashboard`
**Branch:** `main`
**Main file path:** `app.py`
**App URL:** `your-username-sts-claims-dashboard`

### Step 4: Configure Secrets (Optional)
In your Streamlit Cloud app settings, add:
```toml
password = "YourCustomPassword123!"
```

### Step 5: Share Your Dashboard
Your dashboard will be available at:
`https://your-username-sts-claims-dashboard.streamlit.app`

## 🔒 Security Notes
- The dashboard has password protection built-in
- Demo mode is enabled by default for safe testing
- Your actual STS credentials are never stored in the cloud

## 📞 Support
If you need help with deployment, the dashboard includes:
- Built-in demo data for testing
- Export functionality for data analysis
- Comprehensive analytics from your original script
