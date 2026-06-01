# Complete Guide: Converting Images to .PT Dataset and Using It

## Overview
This guide walks you through:
1. **Creating a .pt file** from your image dataset
2. **Loading the .pt file** in your Python code
3. **Using it in ML training** with PyTorch

---

## 📋 Step-by-Step Instructions

### STEP 1: Create Dataset .PT File
**File:** `step1_create_dataset_pt.py`

#### What it does:
- Loads all images from your `dataset/train` and `dataset/valid` folders
- Resizes them to consistent dimensions (416x416 for YOLO)
- Converts images to PyTorch tensors
- Loads associated label files
- Saves everything as a single `.pt` file

#### How to run:
```bash
python step1_create_dataset_pt.py
```

#### What you'll get:
- ✅ `my_dataset.pt` file (contains all images and labels)
- Console output showing:
  - Number of images loaded
  - Tensor shapes
  - File size

#### File structure of .PT:
```
my_dataset.pt contains:
├── 'images': Tensor of shape [N, 3, 416, 416]  (N = total images)
├── 'labels': List of label data for each image
├── 'image_paths': Metadata about image paths
├── 'image_size': [416, 416]
└── 'metadata': Additional info
```

---

### STEP 2: Load and Use the Dataset
**File:** `step2_load_and_use_dataset.py`

#### What it does:
- Demonstrates how to load your `.pt` file
- Creates PyTorch DataLoader for training
- Shows two methods of accessing data

#### How to run:
```bash
python step2_load_and_use_dataset.py
```

#### Two methods to load data:

**METHOD 1: Using DataLoader (RECOMMENDED for training)**
```python
from step2_load_and_use_dataset import load_dataset_with_dataloader

# Load dataset with DataLoader
dataloader, dataset = load_dataset_with_dataloader(
    pt_file_path='./my_dataset.pt',
    batch_size=32,
    shuffle=True
)

# Use in training loop
for batch in dataloader:
    images = batch['image']        # Shape: [batch_size, 3, 416, 416]
    labels = batch['label']        # List of labels
    paths = batch['path']          # Metadata about images
    
    # Your training code here
    # output = model(images)
```

**METHOD 2: Direct Access (for analysis)**
```python
from step2_load_and_use_dataset import load_pt_file_simple

# Load directly
data = load_pt_file_simple('./my_dataset.pt')

# Access components
images = data['images']            # [N, 3, 416, 416]
labels = data['labels']            # List of N labels
image_paths = data['image_paths']  # List of metadata
```

---

### STEP 3: Complete Training Example
**File:** `step3_complete_training_example.py`

#### What it does:
- Shows full training pipeline:
  1. Load dataset from .pt file
  2. Create neural network model
  3. Train the model
  4. Save trained model
  5. Run inference on dataset

#### How to run:
```bash
python step3_complete_training_example.py
```

#### Components included:
- `CustomDataset`: Loads data from .pt file
- `SimpleDetectionModel`: Example CNN architecture
- `Trainer`: Handles training loop
- `InferenceEngine`: Runs predictions

#### Outputs:
- ✅ `trained_model.pth` - Your trained model weights
- Console output showing:
  - Training progress
  - Loss values per epoch
  - Inference results

---

## 🔧 How to Customize for Your Needs

### Adjust Image Size
**In step1_create_dataset_pt.py:**
```python
creator = DatasetCreator(
    dataset_path=DATASET_PATH,
    output_path=OUTPUT_PT_FILE
)
creator.image_size = (640, 640)  # Change from (416, 416) to your size
```

### Change Batch Size
**In step2 or step3:**
```python
dataloader = load_dataset_with_dataloader(
    pt_file_path='./my_dataset.pt',
    batch_size=64,  # Increase or decrease as needed
    shuffle=True
)
```

### Use Different Model Architecture
**In step3_complete_training_example.py:**
Replace `SimpleDetectionModel` with your own model:
```python
# Instead of SimpleDetectionModel
from torchvision.models import resnet50

model = resnet50(pretrained=True)
# Modify final layer for your task
```

### Use GPU
The code automatically detects GPU:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

To force GPU:
```python
device = 'cuda'
```

---

## 🎯 Integration with Your YOLO Code

If you want to use this with YOLO training:

```python
from ultralytics import YOLO
from step2_load_and_use_dataset import load_dataset_with_dataloader

# Load your dataset
dataloader, dataset = load_dataset_with_dataloader('./my_dataset.pt')

# Load YOLO model
model = YOLO('yolov8n.yaml')

# You would typically use YOLO's own training interface:
results = model.train(
    data='data.yaml',      # Your YOLO data config
    epochs=100,
    imgsz=416,
    batch=32,
    device=0              # GPU device
)
```

---

## 📊 Understanding Data Shapes

```
Input Data Structure:
├── images tensor
│   ├── Shape: [N, C, H, W]
│   ├── N = number of images
│   ├── C = 3 (RGB channels)
│   ├── H = 416 (height)
│   └── W = 416 (width)
│
├── labels
│   └── List of N label strings (from .txt files)
│
└── image_paths
    └── List of N metadata dictionaries
```

**Example shapes for 10 images:**
- images: [10, 3, 416, 416]
- labels: 10 items
- image_paths: 10 items

---

## ⚡ Quick Reference

### Create .PT file:
```bash
python step1_create_dataset_pt.py
```

### Load and inspect:
```bash
python step2_load_and_use_dataset.py
```

### Full training example:
```bash
python step3_complete_training_example.py
```

### In your own code:
```python
# Load dataset
from step2_load_and_use_dataset import load_dataset_with_dataloader
dataloader, dataset = load_dataset_with_dataloader('./my_dataset.pt')

# Use in loop
for batch in dataloader:
    images = batch['image']
    # your code here
```

---

## 🐛 Troubleshooting

**Error: "dataset not found"**
- Make sure your folder structure matches:
  ```
  dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  └── valid/
      ├── images/
      └── labels/
  ```

**Error: "CUDA out of memory"**
- Reduce batch_size
- Reduce image_size
- Use CPU instead of GPU

**Error: "No images found"**
- Check file extensions (.jpg, .png, etc.)
- Verify images folder has actual image files

**Large .PT file size**
- This is normal. 1000 images @ 416x416 ≈ 200-300 MB
- Compress if needed using torch.save with compression

---

## 📚 File Reference

| File | Purpose | Run Time |
|------|---------|----------|
| step1_create_dataset_pt.py | Create .pt from images | 1-5 min |
| step2_load_and_use_dataset.py | Load and inspect data | <1 min |
| step3_complete_training_example.py | Full training example | 5-30 min |

---

## 🎓 What You've Learned

✅ How to convert image dataset to .pt format
✅ How to load .pt files in PyTorch
✅ How to create DataLoaders for training
✅ How to build and train models
✅ How to save and load trained models
✅ How to run inference on datasets

---

## 📝 Next Steps

1. Run step1 to create your .pt file
2. Run step2 to verify data loads correctly
3. Run step3 to train a model
4. Customize the code for your specific task
5. Integrate with your YOLO pipeline

Good luck! 🚀
