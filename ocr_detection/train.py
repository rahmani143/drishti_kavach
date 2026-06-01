"""
YOLOv8 Nano Model Training Script for License Plate Detection
Dataset: https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11

This script automates the complete training pipeline:
- Automatically downloads the dataset from Kaggle if not found locally
- Validates dataset structure
- Trains a YOLOv8 nano model on CPU for license plate detection
- Outputs trained weights (best.pt) deployable on Raspberry Pi 4

Requirements:
- ultralytics: pip install ultralytics
- PyYAML: pip install pyyaml
- kaggle: pip install kaggle (for automatic dataset download)
- torch: pip install torch (CPU version for deployment compatibility)

Kaggle Setup:
- Download kaggle.json from https://www.kaggle.com/settings/account
- Place it at ~/.kaggle/kaggle.json
- Run: chmod 600 ~/.kaggle/kaggle.json (on Unix/Linux/Mac)
"""

import os
import sys
import yaml
import zipfile
import shutil
from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Absolute path to the dataset directory
DATASET_DIR = os.path.abspath("C:\\Users\\bss10\\Desktop\\drishti\\drishti_kavach\\archive\\IDDDetectionsYOLODataset")

# Kaggle dataset identifier
KAGGLE_DATASET = "redzapdos123/indian-driving-dataset-detections-yolov11"

# Training hyperparameters
EPOCHS = 10           # Number of training epochs
BATCH_SIZE = 8        # Batch size (CPU-friendly; use 16 if you have sufficient RAM)
IMG_SIZE = 640        # Image size for training
DEVICE = "cpu"        # Training device (CPU-only for Raspberry Pi compatibility)
MODEL = "yolov8n.pt"  # YOLOv8 nano model


# ============================================================================
# DATASET DOWNLOAD AND VALIDATION FUNCTIONS
# ============================================================================

def dataset_exists_and_valid(dataset_path):
    """
    Check if dataset directory exists and contains required subdirectories.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    required_dirs = ["train", "val", "test"]
    required_files = ["data.yaml"]
    
    if not os.path.isdir(dataset_path):
        return False
    
    for subdir in required_dirs:
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            return False
    
    for filename in required_files:
        file_path = os.path.join(dataset_path, filename)
        if not os.path.isfile(file_path):
            return False
    
    return True


def download_dataset_from_kaggle(dataset_id, target_dir):
    """
    Download dataset from Kaggle using the Kaggle Python API.
    
    Requires kaggle.json to be placed at ~/.kaggle/kaggle.json
    
    Args:
        dataset_id (str): Kaggle dataset identifier (owner/dataset-name)
        target_dir (str): Directory to download the dataset into
    """
    try:
        import kaggle
    except ImportError:
        print("❌ Error: kaggle library not found.")
        print("Install it with: pip install kaggle")
        sys.exit(1)
    
    print(f"\n📥 Downloading dataset from Kaggle: {dataset_id}")
    print(f"   Target directory: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        kaggle.api.dataset_download_files(
            dataset_id,
            path=target_dir,
            unzip=False
        )
        print("✓ Dataset download completed.")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure kaggle.json is placed at ~/.kaggle/kaggle.json")
        print("  2. Download it from: https://www.kaggle.com/settings/account")
        print("  3. Set permissions: chmod 600 ~/.kaggle/kaggle.json (Unix/Linux/Mac)")
        sys.exit(1)


def extract_dataset(target_dir):
    """
    Extract ZIP file(s) in the target directory.
    
    Args:
        target_dir (str): Directory containing the ZIP file(s)
    """
    zip_files = [f for f in os.listdir(target_dir) if f.endswith('.zip')]
    
    if not zip_files:
        print("⚠ No ZIP files found to extract.")
        return
    
    for zip_file in zip_files:
        zip_path = os.path.join(target_dir, zip_file)
        print(f"\n📦 Extracting: {zip_file}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            os.remove(zip_path)
            print(f"✓ Extracted and removed: {zip_file}")
        except Exception as e:
            print(f"❌ Failed to extract {zip_file}: {e}")
            sys.exit(1)


def ensure_dataset_available(dataset_dir, kaggle_dataset_id):
    """
    Ensure dataset is available locally. If not, download it from Kaggle.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        kaggle_dataset_id (str): Kaggle dataset identifier
    """
    if dataset_exists_and_valid(dataset_dir):
        print(f"✓ Dataset found and validated at: {dataset_dir}")
        return
    
    print(f"\n⚠ Dataset not found or invalid at: {dataset_dir}")
    print("Initiating automatic download from Kaggle...")
    
    download_dataset_from_kaggle(kaggle_dataset_id, dataset_dir)
    extract_dataset(dataset_dir)
    
    if not dataset_exists_and_valid(dataset_dir):
        print("❌ Dataset validation failed after download.")
        sys.exit(1)
    
    print(f"✓ Dataset is now ready at: {dataset_dir}")


# ============================================================================
# LEGACY VALIDATION FUNCTION (for backward compatibility)
# ============================================================================

def validate_dataset_structure(dataset_root):
    """
    Validate that the dataset has the required structure and all directories exist.
    
    Expected structure:
    - dataset_root/train/images/
    - dataset_root/train/labels/
    - dataset_root/val/images/
    - dataset_root/val/labels/
    - dataset_root/test/images/
    - dataset_root/test/labels/
    - dataset_root/data.yaml
    """
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")
    
    required_dirs = [
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
        "test/images",
        "test/labels",
    ]
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
        if not list(dir_path.glob("*")):
            print(f"Warning: {dir_path} is empty")
    
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    
    print(f"✓ Dataset structure validated at: {dataset_path}")


def update_data_yaml(dataset_root):
    """
    Read the data.yaml file and update paths to use absolute paths.
    This ensures YOLO can find the train, val, and test folders regardless
    of the working directory.
    
    Returns the path to the updated data.yaml file.
    """
    dataset_path = Path(dataset_root)
    data_yaml_path = dataset_path / "data.yaml"
    
    # Read the original data.yaml
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    # Convert paths to absolute paths
    data["path"] = str(dataset_path.absolute())
    data["train"] = str((dataset_path / "train/images").absolute())
    data["val"] = str((dataset_path / "val/images").absolute())
    data["test"] = str((dataset_path / "test/images").absolute())
    
    # Write back the updated data.yaml
    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✓ data.yaml updated with absolute paths")
    print(f"  - Path: {data['path']}")
    print(f"  - Train: {data['train']}")
    print(f"  - Val: {data['val']}")
    print(f"  - Test: {data['test']}")
    
    return str(data_yaml_path)


def train_model(data_yaml_path):
    """
    Load the YOLOv8 nano model and train it on the provided dataset.
    
    Args:
        data_yaml_path: Absolute path to the data.yaml configuration file
    
    Returns:
        Dictionary containing training results and model path information
    """
    print(f"\n{'='*70}")
    print("Starting YOLOv8 Nano Model Training")
    print(f"{'='*70}")
    print(f"Model: {MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Data Config: {data_yaml_path}")
    print(f"{'='*70}\n")
    
    # Load the pretrained YOLOv8 nano model
    model = YOLO(MODEL)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        patience=5,
        save=True,
        verbose=True,
        project="runs/detect",
        name="license_plate_detection",
    )
    
    return results


def print_training_summary(results):
    """
    Print a summary of training results and the location of the best weights.
    """
    # The trained model saves to runs/detect/license_plate_detection/
    run_dir = Path("runs/detect/license_plate_detection")
    best_weights = run_dir / "weights/best.pt"
    last_weights = run_dir / "weights/last.pt"
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    
    if best_weights.exists():
        best_path = best_weights.absolute()
        print(f"✓ Best weights saved at:")
        print(f"  {best_path}")
        print(f"\nTo use the trained model for inference:")
        print(f"  from ultralytics import YOLO")
        print(f"  model = YOLO(r'{best_path}')")
        print(f"  results = model.predict(source='image.jpg')")
    else:
        print("⚠ Best weights file not found. Check training output above.")
    
    if last_weights.exists():
        last_path = last_weights.absolute()
        print(f"\nLast checkpoint saved at:")
        print(f"  {last_path}")
    
    # Print training artifacts location
    artifacts_path = run_dir.absolute()
    print(f"\nAll training artifacts (logs, results, weights) saved at:")
    print(f"  {artifacts_path}")
    print(f"{'='*70}\n")


def main():
    """
    Main training pipeline:
    1. Ensure dataset is available (download from Kaggle if needed)
    2. Validate dataset structure
    3. Update data.yaml with absolute paths
    4. Train the YOLOv8 nano model
    5. Print summary with trained weights location
    """
    try:
        print(f"\n{'='*70}")
        print("YOLOv8 Nano License Plate Detection Training Pipeline")
        print(f"{'='*70}")
        print(f"Dataset Directory: {DATASET_DIR}")
        print(f"Kaggle Dataset: {KAGGLE_DATASET}")
        print(f"{'='*70}\n")
        
        # Ensure dataset is available (download if needed)
        ensure_dataset_available(DATASET_DIR, KAGGLE_DATASET)
        
        # Validate dataset
        validate_dataset_structure(DATASET_DIR)
        
        # Update data.yaml with absolute paths
        data_yaml_path = update_data_yaml(DATASET_DIR)
        
        # Train the model
        results = train_model(data_yaml_path)
        
        # Print training summary
        print_training_summary(results)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nPlease ensure:")
        print(f"  1. DATASET_DIR is set to the correct path")
        print(f"  2. Dataset structure matches the expected format")
        print(f"  3. data.yaml exists in the dataset root")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
