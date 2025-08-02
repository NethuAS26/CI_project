# 🚀 GitHub Pages Deployment Guide

This guide will walk you through deploying your Personality Predictor web app to GitHub Pages.

## 📋 Prerequisites

- ✅ GitHub account
- ✅ Repository with your project files
- ✅ Web app files in the `web-app/` folder

## 🎯 Step-by-Step Deployment

### Step 1: Navigate to Repository Settings

1. **Go to your GitHub repository**: `https://github.com/NethuAS26/CI_project`
2. **Click on the "Settings" tab** (top navigation)
3. **Scroll down to "Pages" section** (left sidebar)

### Step 2: Configure GitHub Pages

1. **Source Selection**:
   - Click on "Deploy from a branch"
   - Select "main" branch (or your default branch)

2. **Folder Selection**:
   - Choose "/ (root)" if you want the app at the root URL
   - OR choose "/web-app" if you want it in a subfolder

3. **Save Configuration**:
   - Click "Save" button
   - Wait for the green checkmark to appear

### Step 3: Access Your Live App

Your app will be available at:
```
https://nethuas26.github.io/CI_project/
```

**Note**: It may take a few minutes for the changes to propagate.

## 🔧 Alternative Deployment Options

### Option A: Deploy from Root Directory

If you want the app at the root URL, move the files:

1. **Move `index.html` to root**:
   ```bash
   mv web-app/index.html ./index.html
   ```

2. **Configure GitHub Pages**:
   - Source: "Deploy from a branch"
   - Branch: "main"
   - Folder: "/ (root)"

3. **Your app will be at**:
   ```
   https://nethuas26.github.io/CI_project/
   ```

### Option B: Deploy from Web-App Folder

Keep the current structure:

1. **Configure GitHub Pages**:
   - Source: "Deploy from a branch"
   - Branch: "main"
   - Folder: "/web-app"

2. **Your app will be at**:
   ```
   https://nethuas26.github.io/CI_project/web-app/
   ```

## 🎨 Custom Domain (Optional)

If you want a custom domain:

1. **In GitHub Pages settings**:
   - Scroll to "Custom domain" section
   - Enter your domain name
   - Click "Save"

2. **Configure DNS**:
   - Add CNAME record pointing to `yourusername.github.io`
   - Wait for DNS propagation (up to 24 hours)

## 🔍 Troubleshooting

### Common Issues

**❌ Page not found (404)**
- Check that the file is named `index.html`
- Ensure the correct folder is selected in GitHub Pages settings
- Wait a few minutes for deployment

**❌ Styling not loading**
- Check browser console for errors
- Ensure all CSS is inline (it is in our app)
- Try hard refresh (Ctrl+F5)

**❌ Form not working**
- Check browser console for JavaScript errors
- Ensure JavaScript is enabled
- Try a different browser

### Debug Steps

1. **Check GitHub Actions**:
   - Go to "Actions" tab in your repository
   - Look for any deployment errors

2. **Check Repository Structure**:
   - Ensure `index.html` is in the correct location
   - Verify file permissions

3. **Test Locally**:
   - Download `index.html`
   - Open in browser to test functionality

## 📱 Testing Your Deployment

### Test Checklist

- ✅ [ ] App loads without errors
- ✅ [ ] Form fields are visible and functional
- ✅ [ ] Submit button works
- ✅ [ ] Results display correctly
- ✅ [ ] Mobile responsive design
- ✅ [ ] No console errors

### Sample Test Data

**Extrovert Test**:
- Time Alone: 2 hours
- Stage Fear: No
- Social Events: 8 per month
- Going Outside: 5 times per week
- Drained After Socializing: No
- Friends Circle: 20 friends
- Post Frequency: 5 posts per week

**Introvert Test**:
- Time Alone: 6 hours
- Stage Fear: Yes
- Social Events: 2 per month
- Going Outside: 2 times per week
- Drained After Socializing: Yes
- Friends Circle: 5 friends
- Post Frequency: 1 post per week

## 🔄 Updating Your App

### Making Changes

1. **Edit the files** in your local repository
2. **Commit and push** to GitHub:
   ```bash
   git add .
   git commit -m "Update web app"
   git push origin main
   ```
3. **GitHub Pages will automatically redeploy** (may take a few minutes)

### Force Redeploy

If changes don't appear:

1. **Go to repository settings**
2. **GitHub Pages section**
3. **Click "Save" again** (even without changes)
4. **Wait for deployment**

## 📊 Monitoring

### Check Deployment Status

1. **Repository Settings → Pages**
2. **Look for green checkmark** indicating successful deployment
3. **Check the deployment time** to see when it was last updated

### View Deployment Logs

1. **Go to "Actions" tab**
2. **Look for "pages build and deployment"**
3. **Click to view detailed logs**

## 🎯 Final Steps

### 1. Test Your Live App

Visit your deployed URL and test all functionality:
- Form submission
- Result display
- Mobile responsiveness
- Different browsers

### 2. Share Your App

Your app is now live and can be shared with:
- Direct URL link
- QR code for mobile access
- Embedded in other websites

### 3. Monitor Performance

- Check page load times
- Monitor user feedback
- Update as needed

## 🚀 Success!

Your Personality Predictor web app is now live on GitHub Pages! 

**Live URL**: `https://nethuas26.github.io/CI_project/`

---

**Happy Deploying! 🎉✨** 