# 🌐 STS Dashboard Deployment Guide

## Option 1: Streamlit Cloud (Recommended - FREE)

### Step 1: Upload to GitHub
1. Create a GitHub account at https://github.com
2. Create a new repository called "sts-dashboard"
3. Upload these files:
   - app.py
   - sts_processor.py
   - requirements.txt
   - README.md

### Step 2: Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your GitHub repository
5. Choose "app.py" as the main file
6. Click "Deploy"
7. Your app will be live at: https://[your-username]-sts-dashboard-app-[random].streamlit.app

**Benefits:**
✅ FREE hosting
✅ Automatic updates when you change code
✅ HTTPS security
✅ Custom domain possible
✅ Password protection included

---

## Option 2: Local Network Sharing

### Quick Setup:
1. Run: `run_dashboard_network.bat`
2. Find your IP address (usually 192.168.x.x)
3. Share URL: `http://YOUR_IP:8501`

**Example:** `http://192.168.1.100:8501`

### To find your IP:
```cmd
ipconfig | findstr "IPv4"
```

**Benefits:**
✅ Works on local network immediately
✅ No internet required
❌ Only works on same network
❌ Computer must stay on

---

## Option 3: ngrok (Public URL from Local)

### Setup:
1. Download ngrok from https://ngrok.com
2. Run your dashboard locally
3. In another terminal: `ngrok http 8501`
4. Share the ngrok URL (e.g., https://abc123.ngrok.io)

**Benefits:**
✅ Instant public URL
✅ Works from anywhere
✅ Free tier available
❌ URL changes each time
❌ Computer must stay on

---

## Option 4: Cloud VPS (Advanced)

### Providers:
- **DigitalOcean:** $5/month droplet
- **AWS EC2:** Free tier available
- **Google Cloud:** $300 free credits

### Setup:
1. Create cloud server
2. Install Python + requirements
3. Run with nginx reverse proxy
4. Set up domain name

**Benefits:**
✅ Always online
✅ Custom domain
✅ Full control
❌ Requires technical setup
❌ Monthly cost

---

## 🔒 Security Features Included

- **Password Protection:** Users must enter password to access
- **Demo Mode:** Safe to share without exposing real credentials
- **No Data Storage:** Processes data in memory only
- **HTTPS:** When deployed to Streamlit Cloud

---

## 📋 Quick Deployment Checklist

### For Streamlit Cloud:
- [ ] Create GitHub account
- [ ] Upload code to GitHub repository
- [ ] Deploy on share.streamlit.io
- [ ] Test the live URL
- [ ] Share URL with team

### For Local Network:
- [ ] Run `run_dashboard_network.bat`
- [ ] Find your IP address
- [ ] Test from another computer
- [ ] Share IP:8501 with team

---

## 🎯 Recommended Approach

**For Delta Air Lines Team:**
1. **Start with Streamlit Cloud** - Free, professional, secure
2. **Custom domain** - Ask IT for subdomain like `sts-analytics.delta.com`
3. **Team access** - Share the public URL with password

**Example final URL:** `https://delta-sts-analytics.streamlit.app`

---

## 💡 Pro Tips

1. **Update Code:** Just push to GitHub, Streamlit Cloud auto-updates
2. **Password Security:** Change password in secrets for production
3. **Custom Branding:** Add Delta logo and colors in CSS
4. **Data Privacy:** All processing happens in user's browser session
5. **Mobile Friendly:** Dashboard works on tablets and phones

---

## 🆘 Support

If you need help with deployment:
1. Streamlit Docs: https://docs.streamlit.io
2. GitHub Help: https://docs.github.com
3. Contact your IT department for domain setup
