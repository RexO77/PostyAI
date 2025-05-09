# Deploying to Hugging Face Spaces

Follow these steps to deploy your Posty AI app to Hugging Face Spaces:

## Prerequisites
1. Create a [Hugging Face account](https://huggingface.co/join)
2. Have your project code in a GitHub repository

## Steps to Deploy

### 1. Create a New Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in the details:
   - Owner: Your username
   - Space name: posty-ai (or any name you prefer)
   - License: Choose an appropriate license
   - Space SDK: Streamlit
   - Space hardware: CPU (Free)
   - Make sure "Public" is selected

### 2. Add Your Groq API Key
1. Go to your new space's "Settings" tab
2. Scroll down to "Repository secrets"
3. Add a new secret:
   - Name: GROQ_API_KEY
   - Value: Your Groq API key

### 3. Connect GitHub Repository
1. In the Settings tab, find "Repository"
2. Click "Import from GitHub"
3. Select your repository containing the Posty AI code
4. Click "Import"

### 4. Verify Deployment
1. Go to the "App" tab of your Space
2. Wait for the build to complete (may take a few minutes)
3. Your app should now be live on Hugging Face Spaces!

## Troubleshooting
- If your app fails to build, check the build logs for errors
- Make sure all required files are present in your repository
- Ensure your Groq API key is properly set up in the repository secrets 