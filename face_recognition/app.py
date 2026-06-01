import os
import cv2
import numpy as np
import pandas as pd
import joblib
from main import identify_face  # Import your face matching function
from risk_assessment import assess_risk_by_name  # Import your risk evaluation function

def run_security_check(image_path):
    """
    Runs the complete pipeline:
    1. Detects and identifies the face from the image.
    2. Retrieves criminal history and predicts risk if known.
    """
    print(f"\n================ Scanning Image: {os.path.basename(image_path)} ================")
    
    # 1. Run Face Recognition Check
    matched_name = identify_face(image_path)
    
    # 2. Process Next Steps Based on Open-Set Identification
    if matched_name is not None:
        print(f"[SYSTEM NOTICE] Match confirmed in face database. Accessing records...")
        # Run the Random Forest Risk Assessment
        assess_risk_by_name(matched_name)
    else:
        print("\n--- Security Status ---")
        print("Evaluation: UNKNOWN PERSON (Not in criminal database)")
        print("Action: No historical record available. Monitor standard behavior.")
    
    print("========================================================================\n")

if __name__ == "__main__":
    # Test Case 1: Test with a known person from your dataset
    # Replace this path with an actual image from one of your actor folders to test a match
    known_test_image = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces\kay_kay_menon\8a30160d68.jpg"
    
    if os.path.exists(known_test_image):
        run_security_check(known_test_image)
    else:
        print("Please provide a valid test image path to run the full pipeline check.")