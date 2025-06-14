#!/usr/bin/env python3
"""
PostyAI Setup Script
Helps users configure their environment and API key
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file with user's API key"""
    env_path = Path('.env')
    
    print("🚀 PostyAI Setup")
    print("================")
    print()
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Setup cancelled.")
            return False
    
    print("To use PostyAI, you need a free Groq API key.")
    print("👉 Get yours at: https://console.groq.com")
    print()
    
    api_key = input("Enter your Groq API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return False
    
    if api_key == "your_groq_api_key_here" or len(api_key) < 10:
        print("❌ Please enter a valid API key.")
        return False
    
    # Create .env file
    env_content = f"""# PostyAI Configuration
GROQ_API_KEY={api_key}

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created successfully!")
        print()
        print("🎉 Setup complete! You can now run:")
        print("   python app.py")
        print()
        print("   Then visit: http://localhost:8000")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import langchain_groq
        import pandas
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function"""
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print("Setting up environment...")
    if not create_env_file():
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 