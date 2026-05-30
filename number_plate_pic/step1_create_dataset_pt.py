"""
STEP 1: Create .pt file from your image dataset
This script loads all images from your dataset folder and saves them as a PyTorch .pt file
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
import json

class DatasetCreator:
    def __init__(self, dataset_path, output_path="dataset.pt"):
        """
        Initialize dataset creator
        
        Args:
            dataset_path: Path to your dataset folder (e.g., './dataset')
            output_path: Where to save the .pt file
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.image_size = (416, 416)  # YOLOv8 standard size (adjust as needed)
        self.images = []
        self.labels = []
        self.image_paths = []
        
    def load_images_and_labels(self):
        """Load all images and their corresponding labels"""
        
        train_images_path = os.path.join(self.dataset_path, 'train', 'images')
        train_labels_path = os.path.join(self.dataset_path, 'train', 'labels')
        valid_images_path = os.path.join(self.dataset_path, 'valid', 'images')
        valid_labels_path = os.path.join(self.dataset_path, 'valid', 'labels')
        
        print("=" * 60)
        print("STEP 1: LOADING DATASET")
        print("=" * 60)
        
        # Load training data
        print("\n📁 Loading TRAINING data...")
        self._load_from_folder(train_images_path, train_labels_path, 'train')
        
        # Load validation data
        print("\n📁 Loading VALIDATION data...")
        self._load_from_folder(valid_images_path, valid_labels_path, 'valid')
        
        print(f"\n✅ Total images loaded: {len(self.images)}")
        return True
    
    def _load_from_folder(self, images_folder, labels_folder, split_name):
        """Load images and labels from a specific folder"""
        
        if not os.path.exists(images_folder):
            print(f"⚠️  {images_folder} not found!")
            return
        
        image_files = sorted([f for f in os.listdir(images_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
        
        print(f"   Found {len(image_files)} images in {split_name}")
        
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(images_folder, img_file)
            label_path = os.path.join(labels_folder, 
                                     Path(img_file).stem + '.txt')
            
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                
                # Resize to consistent size
                img = transforms.Resize(self.image_size)(img)
                
                # Convert to tensor [C, H, W] (channels, height, width)
                img_tensor = transforms.ToTensor()(img)
                
                # Load label if exists
                label_data = None
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        label_data = f.read().strip()
                
                self.images.append(img_tensor)
                self.labels.append(label_data)
                self.image_paths.append({'path': img_path, 'split': split_name})
                
                if (idx + 1) % 10 == 0:
                    print(f"   ✓ Loaded {idx + 1}/{len(image_files)} images", end='\r')
                    
            except Exception as e:
                print(f"   ⚠️  Error loading {img_file}: {e}")
        
        print(f"   ✓ Loaded {len(image_files)} images successfully!     ")
    
    def create_dataset(self):
        """Create tensor dataset and save to .pt file"""
        
        print("\n" + "=" * 60)
        print("STEP 2: CREATING PYTORCH DATASET")
        print("=" * 60)
        
        # Stack all images into single tensor
        print(f"\n🔄 Stacking {len(self.images)} images into tensor...")
        images_tensor = torch.stack(self.images)
        
        print(f"   Shape: {images_tensor.shape}")
        print(f"   Data type: {images_tensor.dtype}")
        print(f"   Min value: {images_tensor.min():.4f}, Max value: {images_tensor.max():.4f}")
        
        # Create dataset dictionary
        dataset = {
            'images': images_tensor,
            'labels': self.labels,
            'image_paths': self.image_paths,
            'image_size': self.image_size,
            'metadata': {
                'total_images': len(self.images),
                'image_shape': images_tensor.shape
            }
        }
        
        return dataset
    
    def save_dataset(self, dataset):
        """Save dataset to .pt file"""
        
        print("\n" + "=" * 60)
        print("STEP 3: SAVING TO .PT FILE")
        print("=" * 60)
        
        output_dir = os.path.dirname(self.output_path) or '.'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n💾 Saving to: {os.path.abspath(self.output_path)}")
        torch.save(dataset, self.output_path)
        
        file_size = os.path.getsize(self.output_path) / (1024 * 1024)  # MB
        print(f"✅ File saved successfully!")
        print(f"   File size: {file_size:.2f} MB")
        
        return True
    
    def verify_dataset(self, pt_file):
        """Verify the saved .pt file"""
        
        print("\n" + "=" * 60)
        print("STEP 4: VERIFYING DATASET")
        print("=" * 60)
        
        print(f"\n🔍 Loading and verifying: {pt_file}")
        loaded_data = torch.load(pt_file)
        
        print(f"\n✅ Verification Results:")
        print(f"   Images tensor shape: {loaded_data['images'].shape}")
        print(f"   Total labels: {len(loaded_data['labels'])}")
        print(f"   Image size: {loaded_data['image_size']}")
        print(f"   Metadata: {loaded_data['metadata']}")
        
        return True
    
    def run(self):
        """Run complete pipeline"""
        try:
            self.load_images_and_labels()
            dataset = self.create_dataset()
            self.save_dataset(dataset)
            self.verify_dataset(self.output_path)
            
            print("\n" + "=" * 60)
            print("✅ DATASET CREATION COMPLETE!")
            print("=" * 60)
            print(f"\n📦 Your .pt file is ready: {self.output_path}")
            print("You can now use this file in your training code!\n")
            
            return True
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "C:\\Users\\bss10\\Desktop\\drishti\\drishti_kavach\\number_plate_pic\\car images"  # Your dataset folder
    OUTPUT_PT_FILE = "C:\\Users\\bss10\\Desktop\\drishti\\drishti_kavach\\number_plate_pic\\dataset.pt"  # Output .pt file"  # Output .pt file
    
    # Create and run dataset creator
    creator = DatasetCreator(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PT_FILE
    )
    
    success = creator.run()
    
    if success:
        print("Next step: Use the generated .pt file with step2_load_and_use_dataset.py")
