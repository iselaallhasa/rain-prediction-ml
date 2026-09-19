

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'pandas', 'numpy', 'lightgbm', 'plotly', 'joblib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {missing}")
        print("Please install using: pip install -r requirements.txt")
        return False
    return True

def main():
    """Launch the Streamlit app"""
    print("=" * 60)
    print("Rain Prediction System - Launching Application")
    print("=" * 60)
    
   
    if not check_dependencies():
        sys.exit(1)
    
   
    model_file = 'lightgbm_rain_model.pkl'
    
    if not os.path.exists(model_file):
        print(f"\nWarning: Model file '{model_file}' not found!")
        print("Please ensure your trained model file is named 'lightgbm_rain_model.pkl'")
        print("and placed in the same directory as this script.")
        
        proceed = input("\nContinue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    else:
        print(f"\nModel file '{model_file}' found successfully!")
    
    print("\nStarting Streamlit application...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the application.\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "ui.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "main":
    main()