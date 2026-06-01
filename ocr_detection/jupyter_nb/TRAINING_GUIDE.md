# YOLOv8 Nano License Plate Detection - Training Guide

## Overview

`train.py` is a production-ready script that automates the complete training pipeline for a **YOLOv8 nano** model optimized for license plate detection on CPU, with deployment target being **Raspberry Pi 4**.

### Key Features

✅ **Automatic Dataset Download** - Downloads from Kaggle if not found locally  
✅ **CPU-Optimized Training** - Explicitly set to `device='cpu'` for compatibility  
✅ **Smart Configuration** - Absolute paths for dataset, 10 epochs default, batch size 8  
✅ **Clean Output** - Production-ready code for Raspberry Pi deployment  
✅ **Flexible** - Easy to customize training parameters at the top of the script  

---

## Installation & Setup

### 1. Install Required Dependencies

```bash
pip install ultralytics pyyaml kaggle torch
```

### 2. Configure Kaggle API

The script uses the Kaggle Python API to automatically download the dataset if not found locally.

**Steps:**
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" - this downloads `kaggle.json`
3. Place the file at:
   - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
4. Set file permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Update Configuration (Optional)

Open `train.py` and adjust the configuration section if needed:

```python
# Absolute path to the dataset directory (relative paths are converted to absolute)
DATASET_DIR = os.path.abspath("IDDDetectionsYOLODataset")

# Training hyperparameters
EPOCHS = 10           # Default: 10 (increase for better accuracy, takes longer)
BATCH_SIZE = 8        # Default: 8 (CPU-friendly; use 16 if sufficient RAM)
IMG_SIZE = 640        # Default: 640 (standard for YOLOv8)
DEVICE = "cpu"        # Always "cpu" for Raspberry Pi compatibility
MODEL = "yolov8n.pt"  # Always "yolov8n.pt" (nano model)
```

---

## Running the Training Script

### Basic Usage

```bash
python train.py
```

### What Happens

1. **Dataset Check** - Verifies if dataset exists locally at `DATASET_DIR`
2. **Auto-Download** (if needed) - Downloads from Kaggle using `kaggle.json`
3. **Extract** - Unzips the downloaded dataset
4. **Validate** - Checks directory structure (train, val, test, data.yaml)
5. **Update Paths** - Modifies `data.yaml` with absolute paths
6. **Train Model** - Starts YOLOv8 nano training on CPU
7. **Print Summary** - Shows final `best.pt` location and inference examples

### Expected Output

```
======================================================================
YOLOv8 Nano License Plate Detection Training Pipeline
======================================================================
Dataset Directory: C:\Users\bss10\Desktop\drishti\drishti_kavach\IDDDetectionsYOLODataset
Kaggle Dataset: redzapdos123/indian-driving-dataset-detections-yolov11
======================================================================

✓ Dataset found and validated at: ...
✓ data.yaml updated with absolute paths
  - Path: ...
  - Train: ...
  - Val: ...
  - Test: ...

======================================================================
Starting YOLOv8 Nano Model Training
======================================================================
Model: yolov8n.pt
Device: cpu
Epochs: 10
Batch Size: 8
Image Size: 640x640
...

======================================================================
Training Complete!
======================================================================
✓ Best weights saved at:
  C:\Users\bss10\Desktop\drishti\drishti_kavach\runs\detect\license_plate_detection\weights\best.pt

To use the trained model for inference:
  from ultralytics import YOLO
  model = YOLO(r'C:\Users\bss10\Desktop\drishti\drishti_kavach\runs\detect\license_plate_detection\weights\best.pt')
  results = model.predict(source='image.jpg')

All training artifacts (logs, results, weights) saved at:
  C:\Users\bss10\Desktop\drishti\drishti_kavach\runs\detect\license_plate_detection
======================================================================
```

---

## Dataset Structure

The script expects the following structure after download/extraction:

```
IDDDetectionsYOLODataset/
├── train/
│   ├── images/           (33,569 images)
│   └── labels/           (33,569 labels - .txt files)
├── val/
│   ├── images/           (4,196 images)
│   └── labels/           (4,196 labels)
├── test/
│   ├── images/           (4,197 images)
│   └── labels/           (4,197 labels)
├── data.yaml             (Updated with absolute paths)
├── license.md
└── ReadMe.md
```

---

## Training Parameters Explained

| Parameter | Default | Notes |
|-----------|---------|-------|
| `EPOCHS` | 10 | Total training passes. Increase to 20-30 for higher accuracy (longer training). |
| `BATCH_SIZE` | 8 | Number of images per batch. Reduce to 4 if OOM errors occur. |
| `IMG_SIZE` | 640 | Input resolution. Standard for YOLOv8. |
| `DEVICE` | "cpu" | **Must remain "cpu"** for Raspberry Pi compatibility. |
| `PATIENCE` | 5 | Early stopping: stop if validation loss doesn't improve for 5 epochs. |

---

## Troubleshooting

### Issue: "kaggle library not found"
**Solution:** Install with `pip install kaggle`

### Issue: "Failed to download dataset"
**Solution:**
1. Verify `kaggle.json` exists at the correct path
2. Check file permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)
3. Test Kaggle access: `kaggle datasets list`

### Issue: Memory errors during training
**Solution:**
- Reduce `BATCH_SIZE` from 8 to 4
- Reduce `IMG_SIZE` from 640 to 416
- Reduce `EPOCHS` for testing

### Issue: Training is very slow
**Expected on CPU.** For 30,000+ images on CPU:
- ~10 epochs: 12-24 hours typical
- For faster prototyping, reduce `EPOCHS` to 5

### Issue: "data.yaml not found"
**Solution:** Ensure dataset is fully extracted. Check that `data.yaml` exists in the dataset root.

---

## Output Files

After training completes, all artifacts are saved to:

```
runs/detect/license_plate_detection/
├── weights/
│   ├── best.pt          ← Use this for inference (Raspberry Pi)
│   └── last.pt
├── results.csv
├── results.png
└── labels/
```

### Key Files

- **`best.pt`** - Best trained weights (lowest validation loss). Use this for inference.
- **`last.pt`** - Last checkpoint (useful if training is interrupted).
- **`results.csv`** - Training metrics (loss, accuracy, etc.).
- **`results.png`** - Training loss/accuracy plots.

---

## Using the Trained Model

### For Inference on Desktop

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO(r'runs/detect/license_plate_detection/weights/best.pt')

# Run inference on an image
results = model.predict(source='test_image.jpg', conf=0.5)

# Get detections
for result in results:
    print(f"Detections: {result.boxes.data}")

# Save annotated result
results[0].save('output.jpg')
```

### For Deployment on Raspberry Pi

1. **Copy `best.pt` to your Pi:**
   ```bash
   scp runs/detect/license_plate_detection/weights/best.pt pi@raspberrypi.local:/home/pi/license_plate_detection/
   ```

2. **On Raspberry Pi, use the same inference code:**
   ```python
   from ultralytics import YOLO
   model = YOLO('best.pt')
   results = model.predict(source='image.jpg')
   ```

---

## Dataset Information

**Source:** https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11

**License:** Check the downloaded dataset for license terms.

**Classes:** License plate detection (single class or multiple if included)

---

## Performance Notes

### Training Speed (Typical)
- **CPU (this script):** ~2-3 hours per epoch (30,000+ images)
- **GPU:** ~10-15 minutes per epoch

### Model Size
- **yolov8n.pt:** ~6.2 MB (nano model)
- **Inference speed:** ~500ms per image on Raspberry Pi 4

---

## Advanced: Customization

### Change Dataset
Edit the configuration:
```python
KAGGLE_DATASET = "owner/your-dataset"
DATASET_DIR = "/path/to/your/dataset"
```

### Use Different YOLOv8 Model
```python
# Nano (default)
MODEL = "yolov8n.pt"  # 3.2M parameters, 6.2 MB

# Small (more accurate, slower)
# MODEL = "yolov8s.pt"  # 11.2M parameters

# Medium (slower still)
# MODEL = "yolov8m.pt"  # 25.9M parameters
```

### Resume Training
If training is interrupted, modify `train.py`:
```python
# Change this:
model = YOLO(MODEL)

# To this (resume from checkpoint):
model = YOLO('runs/detect/license_plate_detection/weights/last.pt')
```

---

## Support & Documentation

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **Kaggle API Docs:** https://www.kaggle.com/settings/account
- **Raspberry Pi Deployment:** https://docs.ultralytics.com/guides/raspberry-pi/

---

## Summary

This script provides a **complete, automated solution** for training a YOLOv8 nano license plate detection model on CPU. It's optimized for eventual deployment on **Raspberry Pi 4** and requires only basic configuration of `kaggle.json`.

Run `python train.py` and the script handles everything from dataset download to training summary!

