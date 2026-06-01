"""
STEP 3: Complete Training Example using .pt Dataset
Shows how to integrate with actual ML model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path


class CustomDataset(Dataset):
    """Load dataset from .pt file"""
    
    def __init__(self, pt_file_path):
        self.data = torch.load(pt_file_path)
        self.images = self.data['images']
        self.labels = self.data['labels']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx]
        }


class SimpleDetectionModel(nn.Module):
    """Simple CNN model for object detection preprocessing"""
    
    def __init__(self):
        super(SimpleDetectionModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # Output single value for detection confidence
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Trainer:
    """Training class to manage model training"""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # Change based on your task
        self.history = {'train_loss': []}
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Dummy target (replace with actual labels)
            targets = torch.rand(outputs.size()).to(self.device)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(self, dataloader, num_epochs=3):
        """Train model for multiple epochs"""
        print("\n" + "=" * 60)
        print("STEP 3: TRAINING MODEL")
        print("=" * 60 + "\n")
        
        for epoch in range(num_epochs):
            print(f"🔄 Epoch {epoch + 1}/{num_epochs}")
            
            avg_loss = self.train_epoch(dataloader)
            self.history['train_loss'].append(avg_loss)
            
            print(f"   Average Loss: {avg_loss:.4f}\n")
        
        return self.history
    
    def save_model(self, save_path):
        """Save trained model"""
        torch.save(self.model.state_dict(), save_path)
        print(f"✅ Model saved to: {save_path}")
    
    def load_model(self, load_path):
        """Load saved model"""
        self.model.load_state_dict(torch.load(load_path))
        print(f"✅ Model loaded from: {load_path}")


class InferenceEngine:
    """Run inference on images using trained model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_batch(self, images):
        """
        Get predictions for a batch of images
        
        Args:
            images: Tensor of shape [batch_size, 3, height, width]
        
        Returns:
            Predictions
        """
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
        
        return outputs
    
    def predict_from_pt(self, pt_file, batch_size=8):
        """Make predictions on all images in .pt file"""
        print("\n" + "=" * 60)
        print("STEP 4: INFERENCE ON DATASET")
        print("=" * 60 + "\n")
        
        dataset = CustomDataset(pt_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image']
            predictions = self.predict_batch(images)
            all_predictions.append(predictions.cpu())
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        
        print(f"\n✅ Inference complete!")
        print(f"   Total predictions: {all_predictions.shape[0]}")
        print(f"   Prediction shape: {all_predictions.shape}")
        
        return all_predictions


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    
    # Configuration
    PT_FILE = "./my_dataset.pt"
    MODEL_SAVE_PATH = "./trained_model.pth"
    BATCH_SIZE = 4
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.001
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device.upper()}")
    
    # Check if .pt file exists
    if not os.path.exists(PT_FILE):
        print(f"\n❌ Error: {PT_FILE} not found!")
        print("Please run step1_create_dataset_pt.py first")
        return
    
    # ===== STEP 1: Load Dataset =====
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60 + "\n")
    
    dataset = CustomDataset(PT_FILE)
    print(f"✅ Dataset loaded: {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    print(f"✅ DataLoader created: {len(dataloader)} batches")
    
    # ===== STEP 2: Create Model =====
    print("\n" + "=" * 60)
    print("STEP 2: CREATING MODEL")
    print("=" * 60 + "\n")
    
    model = SimpleDetectionModel()
    print(f"✅ Model created: {model.__class__.__name__}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== STEP 3: Train Model =====
    trainer = Trainer(model, device=device, learning_rate=LEARNING_RATE)
    history = trainer.train(dataloader, num_epochs=NUM_EPOCHS)
    
    # ===== STEP 4: Save Model =====
    print("\n" + "=" * 60)
    print("STEP 5: SAVING MODEL")
    print("=" * 60 + "\n")
    
    trainer.save_model(MODEL_SAVE_PATH)
    
    # ===== STEP 5: Inference =====
    inference_engine = InferenceEngine(model, device=device)
    predictions = inference_engine.predict_from_pt(PT_FILE, batch_size=BATCH_SIZE)
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("=" * 60)
    print(f"""
Summary:
  ✓ Dataset loaded: {len(dataset)} images
  ✓ Model trained for {NUM_EPOCHS} epochs
  ✓ Model saved to: {MODEL_SAVE_PATH}
  ✓ Inference completed: {len(predictions)} predictions

Files created:
  - {PT_FILE} (Your dataset)
  - {MODEL_SAVE_PATH} (Your trained model)

Next steps:
  1. Use trained model for predictions
  2. Evaluate model performance
  3. Fine-tune hyperparameters
  4. Deploy model for inference
    """)


if __name__ == "__main__":
    main()
