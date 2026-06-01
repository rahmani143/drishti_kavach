"""
STEP 2: Load and use the .pt file in your training code
This script shows how to load and use the dataset.pt file for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os


class CustomDataset(Dataset):
    """Custom Dataset class to load data from .pt file"""
    
    def __init__(self, pt_file_path, transform=None):
        """
        Initialize dataset
        
        Args:
            pt_file_path: Path to your .pt file
            transform: Optional torchvision transforms to apply
        """
        print(f"📂 Loading dataset from: {pt_file_path}")
        
        # Load the .pt file
        self.data = torch.load(pt_file_path)
        
        # Extract components
        self.images = self.data['images']
        self.labels = self.data['labels']
        self.image_paths = self.data['image_paths']
        self.image_size = self.data['image_size']
        
        self.transform = transform
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total images: {len(self.images)}")
        print(f"   Image shape: {self.images.shape}")
        print(f"   Image size: {self.image_size}")
    
    def __len__(self):
        """Return total number of samples"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get single sample"""
        image = self.images[idx]
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'path': image_path
        }


def load_dataset_with_dataloader(pt_file_path, batch_size=32, shuffle=True, num_workers=0):
    """
    Load dataset and create DataLoader for training
    
    Args:
        pt_file_path: Path to .pt file
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader object
    """
    
    print("\n" + "=" * 60)
    print("STEP 2: LOADING DATASET FOR TRAINING")
    print("=" * 60 + "\n")
    
    # Create dataset
    dataset = CustomDataset(pt_file_path)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # For faster GPU transfer
    )
    
    print(f"\n📊 DataLoader Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {len(dataloader)}")
    print(f"   Shuffle: {shuffle}")
    
    return dataloader, dataset


def example_training_loop(dataloader, num_epochs=2):
    """
    Example training loop to show how to use the dataloader
    
    Args:
        dataloader: PyTorch DataLoader
        num_epochs: Number of training epochs
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: EXAMPLE TRAINING LOOP")
    print("=" * 60 + "\n")
    
    for epoch in range(num_epochs):
        print(f"🔄 Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image']
            labels = batch['label']
            paths = batch['path']
            
            # Your training code here
            # Example: model(images), compute loss, backprop, etc.
            
            print(f"   Batch {batch_idx + 1}/{len(dataloader)}")
            print(f"      Image tensor shape: {images.shape}")
            print(f"      Batch size: {images.size(0)}")
            
            # Only show first 3 batches as example
            if batch_idx >= 2:
                print(f"      ... (showing first 3 batches)")
                break
        
        print()


def load_pt_file_simple(pt_file_path):
    """
    Simple way to load .pt file directly
    
    Returns:
        Dictionary with 'images', 'labels', etc.
    """
    data = torch.load(pt_file_path)
    return data


def example_direct_access(pt_file_path):
    """
    Example: Direct access to images without DataLoader
    (useful for analysis or visualization)
    """
    
    print("\n" + "=" * 60)
    print("STEP 4: DIRECT ACCESS EXAMPLE")
    print("=" * 60 + "\n")
    
    # Load directly
    data = load_pt_file_simple(pt_file_path)
    
    images = data['images']
    labels = data['labels']
    
    print(f"📊 Dataset Info:")
    print(f"   Total images: {images.shape[0]}")
    print(f"   Image shape: {images.shape}")
    print(f"   Data type: {images.dtype}")
    print(f"   Value range: [{images.min():.4f}, {images.max():.4f}]")
    
    # Access first image
    first_image = images[0]
    first_label = labels[0]
    
    print(f"\n📌 First Image:")
    print(f"   Shape: {first_image.shape}")
    print(f"   Label: {first_label}")
    
    return data


# ==================== MAIN USAGE EXAMPLES ====================

if __name__ == "__main__":
    
    # Path to your .pt file (generated from step1)
    PT_FILE = "./my_dataset.pt"
    
    # Check if file exists
    if not os.path.exists(PT_FILE):
        print(f"❌ Error: {PT_FILE} not found!")
        print("Please run step1_create_dataset_pt.py first to generate the .pt file")
        exit(1)
    
    # ===== METHOD 1: Using DataLoader (Recommended for Training) =====
    print("\n" + "=" * 60)
    print("METHOD 1: USING DATALOADER")
    print("=" * 60)
    
    dataloader, dataset = load_dataset_with_dataloader(
        pt_file_path=PT_FILE,
        batch_size=4,
        shuffle=True
    )
    
    # Show some batches
    example_training_loop(dataloader, num_epochs=1)
    
    # ===== METHOD 2: Direct Access (For analysis/debugging) =====
    print("\n" + "=" * 60)
    print("METHOD 2: DIRECT ACCESS")
    print("=" * 60)
    
    data = example_direct_access(PT_FILE)
    
    print("\n" + "=" * 60)
    print("✅ DATASET LOADING COMPLETE!")
    print("=" * 60)
    print("""
Now you can use these in your training code:

1. DataLoader approach (Best for training):
   dataloader, dataset = load_dataset_with_dataloader('./my_dataset.pt', batch_size=32)
   for batch in dataloader:
       images = batch['image']
       labels = batch['label']
       # Your training code here

2. Direct access approach:
   data = load_pt_file_simple('./my_dataset.pt')
   images = data['images']
   labels = data['labels']
    """)
