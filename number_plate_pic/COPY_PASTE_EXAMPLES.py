"""
COPY & PASTE CODE EXAMPLES
Quick snippets you can use directly in your own code
"""

# ============================================================================
# EXAMPLE 1: Simplest Way - Load .PT File (3 lines)
# ============================================================================

import torch

# Load your .pt file
data = torch.load('./my_dataset.pt')
images = data['images']    # Shape: [N, 3, 416, 416]
labels = data['labels']    # List of N labels


# ============================================================================
# EXAMPLE 2: Use DataLoader for Training (Recommended)
# ============================================================================

import torch
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.images = data['images']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load dataset
dataset = SimpleDataset('./my_dataset.pt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: {images.shape}")
    # Your training code here


# ============================================================================
# EXAMPLE 3: Training Loop
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 104 * 104, 128)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.images = data['images']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

dataset = SimpleDataset('./my_dataset.pt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model.train()
for epoch in range(5):
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)
        
        # Forward pass
        output = model(images)
        target = torch.randn_like(output)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), './my_model.pth')
print("Model saved!")


# ============================================================================
# EXAMPLE 4: Load Trained Model for Inference
# ============================================================================

import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 104 * 104, 128)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel().to(device)
model.load_state_dict(torch.load('./my_model.pth'))
model.eval()

# Make predictions
data = torch.load('./my_dataset.pt')
images = data['images'].to(device)

with torch.no_grad():
    predictions = model(images)
    print(f"Predictions shape: {predictions.shape}")


# ============================================================================
# EXAMPLE 5: Data Augmentation During Training
# ============================================================================

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class AugmentedDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.images = data['images']
        
        # Define augmentations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

dataset = AugmentedDataset('./my_dataset.pt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(f"Augmented batch: {batch.shape}")


# ============================================================================
# EXAMPLE 6: Batch Processing All Images
# ============================================================================

import torch
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.images = data['images']
        self.labels = data['labels']
        self.paths = data['image_paths']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'path': self.paths[idx]
        }

dataset = SimpleDataset('./my_dataset.pt')
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Process all images
all_results = []
for batch_idx, batch in enumerate(dataloader):
    images = batch['image']
    labels = batch['label']
    paths = batch['path']
    
    print(f"Processing batch {batch_idx}")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {len(labels)}")
    print(f"  Paths: {len(paths)}")
    
    # Process each image in batch
    for i, img in enumerate(images):
        print(f"    Image {i}: {img.shape}")


# ============================================================================
# EXAMPLE 7: Normalize Images
# ============================================================================

import torch
from torchvision import transforms

# Load data
data = torch.load('./my_dataset.pt')
images = data['images']

# Define normalization (ImageNet standard)
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Apply normalization
normalized_images = torch.stack([normalize(img) for img in images])

print(f"Original range: [{images.min():.2f}, {images.max():.2f}]")
print(f"Normalized range: [{normalized_images.min():.2f}, {normalized_images.max():.2f}]")


# ============================================================================
# EXAMPLE 8: Visualize Random Samples
# ============================================================================

import torch
import matplotlib.pyplot as plt

# Load data
data = torch.load('./my_dataset.pt')
images = data['images']

# Show 4 random images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    idx = torch.randint(0, len(images), (1,)).item()
    img = images[idx]
    
    # Convert tensor to numpy for display
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype('uint8')
    
    ax.imshow(img_np)
    ax.set_title(f"Image {idx}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_images.png')
print("Saved sample_images.png")


# ============================================================================
# EXAMPLE 9: Split Dataset for Train/Test
# ============================================================================

import torch
from torch.utils.data import DataLoader, Dataset, random_split

class SimpleDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.images = data['images']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Load full dataset
full_dataset = SimpleDataset('./my_dataset.pt')

# Split: 80% train, 20% test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================================
# EXAMPLE 10: Save & Load Dataset Components Separately
# ============================================================================

import torch

# Load .pt file
data = torch.load('./my_dataset.pt')

# Save only images
torch.save(data['images'], './images_only.pt')

# Save only labels
torch.save(data['labels'], './labels_only.pt')

# Load back
images = torch.load('./images_only.pt')
labels = torch.load('./labels_only.pt')

print(f"Loaded images: {images.shape}")
print(f"Loaded labels: {len(labels)}")


# ============================================================================
# Use these examples in your own code!
# Just copy and paste the section you need
# ============================================================================
