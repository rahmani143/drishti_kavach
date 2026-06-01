import os
import random
import shutil
from pathlib import Path

# Define paths
base_dir = Path(r"C:\Users\bss10\Desktop\drishti\drishti_kavach\datasets\Indian License Plate Detection.yolov8")
train_img_dir = base_dir / "train" / "images"
train_lbl_dir = base_dir / "train" / "labels"

output_dirs = {
    "valid": base_dir / "valid",
    "test": base_dir / "test",
}

# Supported image extensions
img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

# Get all images from the train/images folder
all_images = [f for f in os.listdir(train_img_dir) if Path(f).suffix.lower() in img_extensions]
file_stems = [Path(f).stem for f in all_images]

# Shuffle files randomly using a fixed seed for consistency
random.seed(42)
random.shuffle(file_stems)

# Calculate split counts for 833 files (20% for validation, 10% for test)
total_files = len(file_stems)
valid_count = int(total_files * 0.20)
test_count = int(total_files * 0.10)

# Slice the shuffled list
valid_stems = file_stems[:valid_count]
test_stems = file_stems[valid_count : valid_count + test_count]

splits = {
    "valid": valid_stems,
    "test": test_stems,
}

# Move files to valid and test directories
for split_name, stems in splits.items():
    img_target = output_dirs[split_name] / "images"
    lbl_target = output_dirs[split_name] / "labels"
    img_target.mkdir(parents=True, exist_ok=True)
    lbl_target.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        # Find the original image file with extension
        img_file = next((f for f in all_images if Path(f).stem == stem), None)
        lbl_file = f"{stem}.txt"

        if img_file:
            source_img = train_img_dir / img_file
            source_lbl = train_lbl_dir / lbl_file
            
            # Move image if it exists
            if source_img.exists():
                shutil.move(source_img, img_target / img_file)
            # Move label if it exists
            if source_lbl.exists():
                shutil.move(source_lbl, lbl_target / lbl_file)

# Count remaining training files
remaining_train = len(os.listdir(train_img_dir))

print("Data split complete.")
print(f"Train: {remaining_train} images | Valid: {len(valid_stems)} images | Test: {len(test_stems)} images")