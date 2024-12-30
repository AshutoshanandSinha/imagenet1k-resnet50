import os
import json
import subprocess
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle credentials from environment variables or user input"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        username = input("Enter your Kaggle username: ")
        key = input("Enter your Kaggle API key: ")
        
        credentials = {
            "username": username,
            "key": key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        # Set appropriate permissions
        os.chmod(kaggle_json, 0o600)

def download_imagenet():
    """Download ImageNet-1K dataset from Kaggle competition"""
    # Create data directory
    data_dir = Path('data/imagenet')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ImageNet-1K (ILSVRC2012) dataset...")
    
    # Download competition data
    subprocess.run([
        'kaggle',
        'competitions',
        'download',
        'imagenet-object-localization-challenge',
        '--path',
        str(data_dir)
    ], check=True)
    
    print("\nExtracting dataset (this may take a while)...")
    
    # Unzip the dataset
    subprocess.run([
        'unzip',
        str(data_dir / 'imagenet-object-localization-challenge.zip'),
        '-d',
        str(data_dir)
    ], check=True)
    
    # Clean up zip file
    os.remove(data_dir / 'imagenet-object-localization-challenge.zip')
    
    print("Dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    setup_kaggle_credentials()
    download_imagenet() 