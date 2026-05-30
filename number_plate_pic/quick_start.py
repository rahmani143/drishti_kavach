"""
QUICK START: Run all 3 steps automatically
Execute this to create, load, and train with your dataset in one go
"""

import os
import sys

# Import the step modules
from step1_create_dataset_pt import DatasetCreator
from step2_load_and_use_dataset import load_dataset_with_dataloader
from step3_complete_training_example import Trainer, SimpleDetectionModel, CustomDataset, InferenceEngine
import torch


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def main():
    """Quick start: Run all steps"""
    
    print_banner("🚀 QUICK START: Dataset Creation & Training Pipeline")
    
    # Configuration
    DATASET_PATH = "./dataset"
    PT_FILE = "./my_dataset.pt"
    MODEL_SAVE = "./trained_model.pth"
    BATCH_SIZE = 4
    NUM_EPOCHS = 2
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GPU Available: {torch.cuda.is_available()}")
    print(f"Using Device: {device.upper()}\n")
    
    # ===== STEP 1: Create .PT File =====
    print_banner("STEP 1️⃣ : CREATE .PT FILE FROM IMAGES")
    
    if os.path.exists(PT_FILE):
        print(f"⏭️  Skipping: {PT_FILE} already exists")
        use_existing = input("   Use existing file? (y/n): ").lower()
        if use_existing != 'y':
            print("   Creating new file...")
            creator = DatasetCreator(DATASET_PATH, PT_FILE)
            creator.run()
    else:
        print(f"Creating dataset from: {DATASET_PATH}")
        creator = DatasetCreator(DATASET_PATH, PT_FILE)
        creator.run()
    
    # ===== STEP 2: Load Dataset =====
    print_banner("STEP 2️⃣ : LOAD DATASET FROM .PT FILE")
    
    try:
        dataloader, dataset = load_dataset_with_dataloader(
            pt_file_path=PT_FILE,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total images: {len(dataset)}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Total batches: {len(dataloader)}\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # ===== STEP 3: Create & Train Model =====
    print_banner("STEP 3️⃣ : TRAIN MODEL")
    
    try:
        model = SimpleDetectionModel()
        print(f"✅ Model created: {model.__class__.__name__}")
        
        trainer = Trainer(model, device=device, learning_rate=0.001)
        print(f"✅ Trainer initialized\n")
        
        history = trainer.train(dataloader, num_epochs=NUM_EPOCHS)
        
        # Save model
        trainer.save_model(MODEL_SAVE)
        print(f"\n✅ Model saved to: {MODEL_SAVE}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # ===== STEP 4: Run Inference =====
    print_banner("STEP 4️⃣ : RUN INFERENCE")
    
    try:
        inference_engine = InferenceEngine(model, device=device)
        predictions = inference_engine.predict_from_pt(PT_FILE, batch_size=BATCH_SIZE)
        print(f"✅ Inference complete: {len(predictions)} predictions generated")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return
    
    # ===== Final Summary =====
    print_banner("✅ PIPELINE COMPLETE!")
    
    print("""
📊 Summary of Generated Files:
    ✓ my_dataset.pt          - Your dataset file
    ✓ trained_model.pth      - Your trained model weights
    
📈 Results:
    ✓ Dataset: Successfully loaded
    ✓ Training: Completed for """ + str(NUM_EPOCHS) + """ epochs
    ✓ Inference: All images processed
    
🎯 Next Steps:
    1. Check the generated files
    2. Modify step3_complete_training_example.py for your task
    3. Adjust hyperparameters (learning_rate, batch_size, epochs)
    4. Use different model architectures
    5. Integrate with your YOLO pipeline
    
💡 Tips:
    - Edit step1_create_dataset_pt.py to change image size
    - Edit step3_complete_training_example.py to use custom models
    - Run individual steps separately if needed
    - Use step2_load_and_use_dataset.py for data exploration
    
📚 Full documentation in: README_DATASET_GUIDE.md
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
