# Jupyter Notebook Training Guide

## 📓 train.ipynb - Properly Structured Notebook

Your `train.ipynb` is a **complete, production-ready Jupyter notebook** for training YOLOv8 nano license plate detection.

### What Makes It Different from train.py

| Aspect | train.py | train.ipynb |
|--------|----------|-----------|
| Format | Single Python script | 30 organized Jupyter cells |
| Usage | Run once: `python train.py` | Run cells sequentially in Jupyter |
| Interactivity | Linear execution | Can modify & re-run cells |
| Best for | Automation, servers | Interactive exploration, experimentation |
| Cell Structure | N/A | Markdown + Code cells |
| Breakpoints | N/A | Natural breaks between logical steps |

---

## 🎯 Notebook Structure

The notebook is organized into **10 logical steps**:

### 1️⃣ Install Dependencies
```
!pip install ultralytics pyyaml kaggle torch -q
```
Installs all required libraries in one cell.

### 2️⃣ Import Libraries
```python
import os, sys, yaml, zipfile, Path
from ultralytics import YOLO
```
Imports all required modules with verification.

### 3️⃣ Configuration
```python
DATASET_DIR = os.path.abspath('IDDDetectionsYOLODataset')
EPOCHS = 10
BATCH_SIZE = 8
IMG_SIZE = 640
DEVICE = 'cpu'
MODEL = 'yolov8n.pt'
```
Edit these settings as needed. All in one easy-to-modify cell.

### 4️⃣-5️⃣ Define Helper Functions
- **Dataset Download** - Kaggle API integration
- **Dataset Extraction** - ZIP file handling
- **Dataset Validation** - Structure verification
- **YAML Update** - Path management

Each function group in separate, runnable cells.

### 6️⃣ Prepare Dataset
```
Dataset Check → Download if Missing → Validate → Ready
```
One cell handles the entire dataset preparation pipeline.

### 7️⃣ Update YAML Configuration
Update data.yaml with absolute paths for train/val/test folders.

### 8️⃣ Start Training
Runs the main training loop with full output streaming.

### 9️⃣ Test Inference
Load the trained model and run predictions on test images.

### 🔟 Deploy to Raspberry Pi
Instructions and code for copying the model to your Pi.

---

## ✨ Key Features

### ✅ Markdown Documentation
Each section has clear markdown explanations:
- What the step does
- Expected duration
- How to customize parameters

### ✅ Modular Cells
Each code cell is independent and can be re-run:
```
Cell 1: Install
Cell 2: Import
Cell 3: Configure (edit this)
Cell 4: Define functions
...
Cell N: Train (run this when ready)
```

### ✅ Progress Feedback
Each cell prints clear status messages:
```
✓ Dependencies installed
✓ All imports successful
✓ Configuration loaded
✓ Download functions defined
...
✓ Dataset ready for training
```

### ✅ Interactive Configuration
Edit the configuration cell (Step 3) to customize:
- Dataset location
- Number of epochs
- Batch size
- Training device

Changes apply to all subsequent cells automatically.

### ✅ Safety & Recovery
- Cells can be re-run independently
- Training saves checkpoints (can resume)
- No data loss if interrupted
- Clear error messages with solutions

---

## 🚀 How to Use

### Open in Jupyter Notebook
```bash
jupyter notebook train.ipynb
```

### Open in Jupyter Lab
```bash
jupyter lab train.ipynb
```

### Open in VS Code
1. Open VS Code
2. Open folder: `C:\Users\bss10\Desktop\drishti\drishti_kavach`
3. Right-click `train.ipynb` → "Open with" → "Notebook"

### Open in Google Colab (Cloud)
1. Upload to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Run cells sequentially

---

## 📊 Running the Notebook

### Step-by-Step Workflow

```
1. Open train.ipynb in Jupyter/Lab/VS Code
2. Run Cell 1: Install Dependencies (first time only)
3. Run Cell 2: Import Libraries
4. Run Cell 3: Configuration (EDIT THIS IF NEEDED)
5. Run Cell 4: Define Download Functions
6. Run Cell 5: Define Validation Functions
7. Run Cell 6: Define YAML Functions
8. Run Cell 7: Define Training Functions
9. Run Cell 8: Prepare Dataset (auto-downloads if needed)
10. Run Cell 9: Update YAML Configuration
11. Run Cell 10: Start Training (longest step - 12-24 hours on CPU)
12. Run Cell 11: Test Inference
13. Run Cell 12: Deploy to Raspberry Pi
```

### Expected Output After Each Cell

| Cell | Output |
|------|--------|
| 1 | `✓ Dependencies installed` |
| 2 | `✓ All imports successful` |
| 3 | `Configuration: Dataset: ..., Epochs: 10, ...` |
| 4 | `✓ Download functions defined` |
| 5 | `✓ Validation functions defined` |
| 6 | `✓ YAML update function defined` |
| 7-9 | Function definitions confirmed |
| 10 | `✓ Dataset found` or `📥 Downloading...` |
| 11 | `✓ data.yaml updated with absolute paths` |
| 12 | `🔄 Training progress...` → `✅ TRAINING COMPLETED!` |
| 13 | `✓ Model loaded: best.pt` |

---

## 🎛️ Customization Examples

### Example 1: Change Number of Epochs
Edit **Cell 3 (Configuration)**:
```python
# Before
EPOCHS = 10

# After - for higher accuracy (takes longer)
EPOCHS = 30
```
Then re-run cells 10-12.

### Example 2: Reduce Batch Size (If OOM)
Edit **Cell 3 (Configuration)**:
```python
# Before
BATCH_SIZE = 8

# After - use smaller batches
BATCH_SIZE = 4
```
Then re-run cells 10-12.

### Example 3: Different Dataset Directory
Edit **Cell 3 (Configuration)**:
```python
# Before
DATASET_DIR = os.path.abspath('IDDDetectionsYOLODataset')

# After - custom path
DATASET_DIR = '/mnt/external_drive/datasets/licenses'
```
Then re-run cells 8-12.

---

## ⏸️ Pausing & Resuming

### Stop Training
- Press `Ctrl+C` in the notebook cell

### Resume Training
1. Comment out the download cells (Cell 8)
2. Run Cell 10 (Start Training)
3. It will detect the checkpoint and continue

Or use the last checkpoint directly:
```python
model = YOLO('runs/detect/license_plate_detection/weights/last.pt')
results = model.train(...)  # Continues from last epoch
```

---

## 📝 Output Files After Training

```
runs/detect/license_plate_detection/
├── weights/
│   ├── best.pt          ← USE THIS for inference
│   └── last.pt          ← For resuming training
├── results.csv
├── results.png
├── confusion_matrix.png
└── labels/
```

**best.pt** is your trained model ready for:
- Desktop testing
- Raspberry Pi deployment
- Production inference

---

## 🔄 Notebook vs Script

| Scenario | Use Notebook | Use Script |
|----------|---|---|
| First time training | ✅ Notebook | Script |
| Learning/exploring | ✅ Notebook | - |
| Modifying parameters | ✅ Notebook | Script |
| Automation/CI/CD | - | ✅ Script |
| Server training | - | ✅ Script |
| Interactive testing | ✅ Notebook | - |
| Production deployment | - | ✅ Script |

---

## 🆘 Troubleshooting

### Notebook won't open
- Install Jupyter: `pip install jupyter`
- Or use VS Code with Python extension

### Cells won't execute
- Ensure you ran Cell 1 (Install Dependencies)
- Run cells in order (don't skip cells)

### Dataset download fails
- Verify `~/.kaggle/kaggle.json` exists
- Test with: `kaggle datasets list`

### Training runs out of memory
- Reduce `BATCH_SIZE` to 4 in Cell 3
- Reduce `EPOCHS` for testing
- Check system RAM usage

### Can't find best.pt after training
- Check Cell 12 output for the path
- Default: `runs/detect/license_plate_detection/weights/best.pt`

---

## 📚 Related Files

- **train.py** - Command-line script version (automation)
- **README_TRAIN.md** - Comprehensive training guide
- **TRAINING_GUIDE.md** - Detailed hyperparameter reference
- **QUICK_START.txt** - Quick reference card

---

## 🎯 Summary

Your `train.ipynb` is a **well-structured, interactive notebook** perfect for:
- 🔬 Interactive model training
- 🧪 Experimenting with parameters
- 📊 Monitoring training progress in real-time
- 🎓 Learning the training pipeline
- 🚀 Quick prototyping

It includes **automatic Kaggle dataset download**, **CPU-optimized training**, and **complete Raspberry Pi deployment instructions**.

Simply open and run cells sequentially! 🚀

