import os
import json
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm

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
        
        os.chmod(kaggle_json, 0o600)

def extract_imagenet(data_dir, zip_path):
    """Extract and organize ImageNet dataset from competition zip file"""
    print(f"Starting extraction process...")
    print(f"Zip file location: {zip_path}")
    print(f"Target directory: {data_dir}")
    
    # Verify zip file exists
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found at {zip_path}")
    
    # Create necessary directories
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Extract main zip file
    print(f"Extracting main competition zip file...")
    try:
        # Change to the data directory before extracting
        os.chdir(str(data_dir))
        result = subprocess.run([
            'unzip', 
            str(zip_path.name),  # Use just the filename since we're in the correct directory
            '-d', 
            '.'  # Extract to current directory
        ], check=True, capture_output=True, text=True)
        print("Extraction successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise

    # Move and organize files
    ilsvrc_dir = data_dir / 'ILSVRC'
    if ilsvrc_dir.exists():
        print("Organizing dataset structure...")
        
        # Move training data
        src_train = ilsvrc_dir / 'Data' / 'CLS-LOC' / 'train'
        if src_train.exists():
            print("Moving training data...")
            for item in src_train.iterdir():
                if item.is_dir():
                    shutil.move(str(item), str(train_dir))
        else:
            print(f"Warning: Training directory not found at {src_train}")
        
        # Move validation data
        src_val = ilsvrc_dir / 'Data' / 'CLS-LOC' / 'val'
        if src_val.exists():
            print("Moving validation data...")
            for item in src_val.iterdir():
                shutil.move(str(item), str(val_dir))
        else:
            print(f"Warning: Validation directory not found at {src_val}")
        
        # Clean up extracted files
        print("Cleaning up...")
        shutil.rmtree(str(ilsvrc_dir))
    else:
        print(f"Warning: Expected directory structure not found at {ilsvrc_dir}")
    
    print("Dataset organization completed!")

def download_imagenet():
    """Organize ImageNet-1K dataset files"""
    data_dir = Path('data/imagenet')
    data_dir = data_dir.resolve()  # Get absolute path
    
    # Look for the zip file in the data directory
    zip_file = data_dir / 'imagenet-object-localization-challenge.zip'
    
    print(f"Checking for ImageNet zip file at: {zip_file}")
    
    if zip_file.exists():
        print(f"\nFound ImageNet zip file at {zip_file}")
        print(f"File size: {zip_file.stat().st_size / (1024*1024*1024):.2f} GB")
        extract_imagenet(data_dir, zip_file)
    else:
        print("\nERROR: Could not find 'imagenet-object-localization-challenge.zip'")
        print(f"Please ensure the file exists at: {zip_file}")

if __name__ == "__main__":
    try:
        download_imagenet()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc() 