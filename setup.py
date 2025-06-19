"""
Setup Script for Customer Churn Prediction ML Project
BUS8405 Machine Learning Assignment Implementation
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for the ML project"""
    print("📦 Installing Required Packages for ML Project...")
    print("="*50)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def verify_installation():
    """Verify that all packages are installed correctly"""
    print("\n🔍 Verifying Installation...")
    print("="*30)
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'plotly', 'xgboost', 'lightgbm', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        return False
    else:
        print("\n✅ All packages verified!")
        return True

def setup_project_structure():
    """Ensure project directory structure exists"""
    print("\n📁 Setting up Project Structure...")
    print("="*35)
    
    directories = [
        'data',
        'data/raw',
        'src',
        'models',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/")
    
    print("\n✅ Project structure ready!")

def main():
    """Main setup function"""
    print("🚀 Customer Churn Prediction - Project Setup")
    print("="*50)
    
    # Setup directory structure
    setup_project_structure()
    
    # Install requirements
    if install_requirements():
        # Verify installation
        if verify_installation():
            print("\n🎉 Setup Complete!")
            print("="*20)
            print("\n📋 Next Steps:")
            print("1. Download dataset: python download_data.py")
            print("2. Run analysis: python main.py")
            print("\n✨ Ready to start your ML assignment!")
        else:
            print("\n❌ Setup incomplete. Please check package installation.")
    else:
        print("\n❌ Failed to install packages. Please check your pip installation.")

if __name__ == "__main__":
    main()
