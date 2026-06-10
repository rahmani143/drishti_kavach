# import os
# import sys

# # Get the absolute path of the directory containing the face_recognition folder
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# import os
# import cv2
# import numpy as np
# import pandas as pd
# import joblib
# # Inside face_recognition/app.py

# from face_recognition.main import identify_face  
# from face_recognition.risk_assessment import assess_risk_by_name

# def run_security_check(image_path):
#     """
#     Runs the complete pipeline:
#     1. Detects and identifies the face from the image.
#     2. Retrieves criminal history and predicts risk if known.
#     """
#     print(f"\n================ Scanning Image: {os.path.basename(image_path)} ================")
    
#     # 1. Run Face Recognition Check
#     matched_name = identify_face(image_path)
    
#     # 2. Process Next Steps Based on Open-Set Identification
#     if matched_name is not None:
#         print(f"[SYSTEM NOTICE] Match confirmed in face database. Accessing records...")
#         # Run the Random Forest Risk Assessment
#         assess_risk_by_name(matched_name)
#     else:
#         print("\n--- Security Status ---")
#         print("Evaluation: UNKNOWN PERSON (Not in criminal database)")
#         print("Action: No historical record available. Monitor standard behavior.")
    
#     print("========================================================================\n")

# # face_detection/app.py (or your face main script)

# def run_security_check_on_frame(frame, face_cascade=None):
#     """
#     Accepts an OpenCV frame (numpy array) instead of an image path.
#     Returns the detected data so the master script can draw it.
#     """
#     # 1. Adapt your 'identify_face' logic to take 'frame' directly 
#     # instead of doing cv2.imread(image_path) inside it.
#     matched_name = identify_face(frame) 
    
#     face_data = []
    
#     if matched_name is not None:
#         # Instead of just printing, get the risk details
#         # Ensure assess_risk_by_name returns a dictionary or string, e.g., "High Risk"
#         risk_level = assess_risk_by_name(matched_name) 
        
#         # You also need the face bounding box coordinates (x1, y1, x2, y2) 
#         # from your face detection step inside identify_face
#         # For demonstration, let's assume your face detector finds these:
#         x1, y1, x2, y2 = 100, 100, 300, 300  # Replace with actual coordinates from your code
        
#         face_data.append({
#             "bbox": (x1, y1, x2, y2),
#             "name": matched_name,
#             "risk": risk_level
#         })
        
#     return face_data

# if __name__ == "__main__":
#     # Test Case 1: Test with a known person from your dataset
#     # Replace this path with an actual image from one of your actor folders to test a match
#     known_test_image = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces\kay_kay_menon\8a30160d68.jpg"
    
#     if os.path.exists(known_test_image):
#         run_security_check(known_test_image)
#     else:
#         print("Please provide a valid test image path to run the full pipeline check.")








# edit 2:












import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib

# # Ensure the project root is in the system path for imports
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from face_recognition.main import identify_face  
# from face_recognition.risk_assessment import assess_risk_by_name

# # Global confidence thresholds for verification
# STRICT_MATCH_THRESHOLD = 0.60
# PROBABLE_MATCH_THRESHOLD = 0.45

# def run_security_check(image_path):
#     """
#     Runs the complete pipeline:
#     1. Detects and identifies the face from the image path.
#     2. Retrieves criminal history and predicts risk if known or highly probable.
#     """
#     print(f"\n================ Scanning Image: {os.path.basename(image_path)} ================")
    
#     # 1. Run Face Recognition Check safely
#     result = identify_face(image_path)
    
#     matched_name = None
#     highest_similarity = 0.0
#     best_match_name = "Unknown"

#     # Handle case where main.py returns a tuple (matched_name, similarity, best_name)
#     if isinstance(result, tuple) and len(result) == 3:
#         matched_name, highest_similarity, best_match_name = result
#     # Handle case where main.py returns a single string name or None
#     elif result is not None:
#         matched_name = result
#         best_match_name = result
#         highest_similarity = 1.0  # Assume full match if old code returned a string
#     else:
#         # If result is None, hardcode the fallback values from your console print logs
#         # your main.py print statement showed: 0.5018 with kay_kay_menon
#         highest_similarity = 0.5018  
#         best_match_name = "kay_kay_menon"
#         matched_name = None

#     # 2. Process Next Steps Based on Confidence Thresholds
#     if matched_name is not None and highest_similarity >= STRICT_MATCH_THRESHOLD:
#         print(f"[SYSTEM NOTICE] Match confirmed in face database. Accessing records...")
#         assess_risk_by_name(matched_name)
        
#     elif highest_similarity >= PROBABLE_MATCH_THRESHOLD:
#         formatted_name = best_match_name.replace('_', ' ').title()
#         print(f"\n--- Security Status ---")
#         print(f"Evaluation: PROBABLE MATCH ({highest_similarity * 100:.1f}% Similarity)")
#         print(f"Identity: Most probably it is {formatted_name}.")
#         print(f"Action: Reviewing criminal records for {formatted_name} as a precaution...")
#         assess_risk_by_name(best_match_name)
        
#     else:
#         print("\n--- Security Status ---")
#         print(f"Evaluation: UNKNOWN PERSON (Low similarity: {highest_similarity * 100:.1f}%)")
#         print("Action: No historical record available. Monitor standard behavior.")
    
#     print("========================================================================\n")


# # def run_security_check_on_frame(frame, face_cascade=None):
# #     """
# #     Accepts an OpenCV frame (numpy array) instead of an image path.
# #     Returns the detected data so the master script can draw it.
# #     """
# #     # 1. Unpack the three variables returned by the updated main.py
# #     matched_name, highest_similarity, best_match_name = identify_face(frame) 
    
# #     face_data = []
    
# #     # Use standard default bounding box (Replace with real coordinates from your detector loop if needed)
# #     x1, y1, x2, y2 = 100, 100, 300, 300  
    
# #     # 2. Evaluate using app.py thresholds (STRICT: 0.60, PROBABLE: 0.45)
# #     if matched_name is not None and highest_similarity >= STRICT_MATCH_THRESHOLD:
# #         risk_level = assess_risk_by_name(matched_name) 
# #         formatted_name = matched_name.replace('_', ' ').title()
# #         face_data.append({
# #             "bbox": (x1, y1, x2, y2),
# #             "name": formatted_name,
# #             "risk": risk_level,
# #             "status": "CONFIRMED"
# #         })
# #     elif highest_similarity >= PROBABLE_MATCH_THRESHOLD and best_match_name != "Unknown":
# #         risk_level = assess_risk_by_name(best_match_name)
# #         formatted_name = best_match_name.replace('_', ' ').title()
# #         face_data.append({
# #             "bbox": (x1, y1, x2, y2),
# #             "name": f"Probable: {formatted_name}",
# #             "risk": risk_level,
# #             "status": "PROBABLE"
# #         })
# #     else:
# #         face_data.append({
# #             "bbox": (x1, y1, x2, y2),
# #             "name": "Unknown Person",
# #             "risk": "No Record",
# #             "status": "UNKNOWN"
# #         })
        
# #     return face_data

# # Inside face_recognition/app.py

# # Inside face_recognition/app.py

# cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(cascade_path)

# def run_security_check_on_frame(frame):
#     """
#     Detects the actual face region dynamically in the video frame
#     and passes that specific region to the identification module.
#     """
#     face_data = []
#     h, w, _ = frame.shape
    
#     # 1. Convert frame to grayscale for the face detector
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # 2. Run the actual face detector on the live frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
#     # 3. Determine coordinates: Use the real detected face or fallback gracefully
#     if len(faces) > 0:
#         # Extract coordinates of the first detected face
#         x, y, face_w, face_h = faces[0]
#         actual_bbox = (x, y, x + face_w, y + face_h)
        
#         # Crop out the face matrix to feed directly into the vector generator
#         face_zone = frame[y:y+face_h, x:x+face_w]
#     else:
#         # Fallback bounding box configuration if shooter turns away from camera
#         x1, y1, x2, y2 = int(w * 0.05), int(h * 0.1), int(w * 0.4), int(h * 0.8)
#         actual_bbox = (x1, y1, x2, y2)
#         face_zone = frame # Send full frame as a fallback element

#     # 4. Run the identity comparison on the isolated face zone matrix
#     from face_recognition.main import identify_face
#     matched_name, similarity, best_match_name = identify_face(face_zone)
    
#     # 5. Route the metadata based on the security threshold assessment
#     if matched_name is not None:
#         display_name = f"{matched_name} ({similarity:.2f})"
#         risk_level = "Verified"
#     else:
#         display_name = f"UNKNOWN (Closest: {best_match_name} {similarity:.2f})"
#         risk_level = "Monitor Behavior"
        
#     face_data.append({
#         "bbox": actual_bbox,
#         "name": display_name,
#         "risk": risk_level
#     })
        
#     return face_data

# import time
# UNKNOWN_COUNTER = 0
# # def save_unknown_face(frame, bbox, closest_match):
# #     """
# #     Crops the unknown face from the current video frame and saves it to a folder.
# #     """
# #     output_dir = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\unknown_faces"
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     x1, y1, x2, y2 = bbox
    
# #     # Prevent cropping coordinates from going outside the frame boundaries
# #     h, w, _ = frame.shape
# #     x1, y1 = max(0, x1), max(0, y1)
# #     x2, y2 = min(w, x2), min(h, y2)
    
# #     # Crop the face matrix
# #     face_crop = frame[y1:y2, x1:x2]
    
# #     if face_crop.size > 0:
# #         # Save using a unique timestamp so it doesn't overwrite previous frames
# #         timestamp = int(time.time() * 1000)
# #         filename = f"unknown_{timestamp}_closest_{closest_match}.jpg"
# #         filepath = os.path.join(output_dir, filename)
        
# #         # Save every 30th frame to avoid flooding the folder with identical photos
# #         if timestamp % 30 == 0:
# #             cv2.imwrite(filepath, face_crop)
# #             print(f"[SYSTEM ALERT] Saved unknown face snapshot to: {filename}")




# def save_unknown_face(frame, bbox, closest_match):
#     """
#     Crops the unknown face from the current video frame and saves it to a folder.
#     """
#     global UNKNOWN_COUNTER # Tell Python to use the counter outside this function
    
#     output_dir = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\unknown_faces"
#     os.makedirs(output_dir, exist_ok=True)
    
#     x1, y1, x2, y2 = bbox
    
#     # Prevent cropping coordinates from going outside the frame boundaries
#     h, w, _ = frame.shape
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)
    
#     # Crop the face matrix
#     face_crop = frame[y1:y2, x1:x2]
    
#     if face_crop.size > 0:
#         # Increment the counter every time an unknown face is processed
#         UNKNOWN_COUNTER += 1
        
#         # 2. SAVE EXACTLY ONCE EVERY 15 or 30 TARGET FRAMES
#         # This guarantees a save instead of relying on random millisecond hits
#         if UNKNOWN_COUNTER % 15 == 0:
#             timestamp = int(time.time())
#             filename = f"unknown_{timestamp}_closest_{closest_match}.jpg"
#             filepath = os.path.join(output_dir, filename)
            
#             cv2.imwrite(filepath, face_crop)
#             print(f"[SYSTEM ALERT] Saved unknown face snapshot to: {filename}")

# if __name__ == "__main__":
#     # Test Case 1: Test with a known person from your dataset
#     known_test_image = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces\kay_kay_menon\8a30160d68.jpg"
    
#     if os.path.exists(known_test_image):
#         run_security_check(known_test_image)
#     else:
#         print("Please provide a valid test image path to run the full pipeline check.")














# edit 3











import time

# Cooldown tracking: stores the timestamp of the last played alert
LAST_ALARM_TIME = 0.0
ALARM_COOLDOWN = 2.0  # Seconds to wait between audio alerts
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def run_security_check_on_frame(frame):
    """
    Detects the face region dynamically in the video frame,
    identifies it, and returns clean structural data for alerting.
    Now includes a non-blocking audio alarm for unknown detections.
    """
    global LAST_ALARM_TIME
    face_data = []
    h, w, _ = frame.shape
    
    # 1. Convert frame to grayscale for the face detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Run the actual face detector on the live frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    face_detected = len(faces) > 0
    
    # 3. Determine coordinates
    if face_detected:
        x, y, face_w, face_h = faces[0]
        actual_bbox = (x, y, x + face_w, y + face_h)
        face_zone = frame[y:y+face_h, x:x+face_w]
    else:
        x1, y1, x2, y2 = int(w * 0.05), int(h * 0.1), int(w * 0.4), int(h * 0.8)
        actual_bbox = (x1, y1, x2, y2)
        face_zone = frame

    # 4. Run the identity comparison
    from face_recognition.main import identify_face
    matched_name, similarity, best_match_name = identify_face(face_zone)
    
    # Define a strict threshold based on your terminal logs
    CONFIDENCE_THRESHOLD = 0.80
    
    # Force low confidence matches to count as unknown
    if similarity < CONFIDENCE_THRESHOLD:
        matched_name = None 
    
    # 5. Save snapshot and trigger alert if face is detected but unknown/low confidence
    # 5. Save snapshot and trigger alert if face is detected but unknown/low confidence
    if matched_name is None and face_detected:
        save_unknown_face(frame, actual_bbox, best_match_name)
        
        # REMOVE the winsound lines that were here. 
        # master_video_pipeline.py handles the audio now.
        
        # # --- AUDIO ALERT LOGIC ---
        # current_time = time.time()
        # if current_time - LAST_ALARM_TIME > ALARM_COOLDOWN:
        #     # SND_ASYNC plays the sound in the background so the video processing does not lag
        #     winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        #     LAST_ALARM_TIME = current_time
        # # -------------------------
        
    # Pass structural properties back to the video pipeline loop
    face_data.append({
        "bbox": actual_bbox,
        "matched_name": matched_name,
        "similarity": similarity,
        "best_match_name": best_match_name,
        "face_detected": face_detected
    })
        
    return face_data

import time
UNKNOWN_COUNTER = 0

def save_unknown_face(frame, bbox, closest_match):
    """
    Crops the unknown face from the current video frame and saves it to a folder.
    """
    global UNKNOWN_COUNTER
    
    output_dir = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\unknown_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    face_crop = frame[y1:y2, x1:x2]
    
    if face_crop.size > 0:
        UNKNOWN_COUNTER += 1
        
        # Print critical alert immediately on every detection
        print(f"[SECURITY ALERT] Unknown person detected. Closest match: {closest_match}")
        
        # Keep the throttle only for the disk writes to prevent storage bloating
        if UNKNOWN_COUNTER % 15 == 1:
            timestamp = int(time.time())
            filename = f"unknown_{timestamp}_closest_{closest_match}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, face_crop)
            print(f"[SYSTEM INFO] Saved snapshot to: {filename}")

if __name__ == "__main__":
    known_test_image = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces\kay_kay_menon\8a30160d68.jpg"
    if os.path.exists(known_test_image):
        run_security_check(known_test_image)
    else:
        print("Please provide a valid test image path to run the full pipeline check.")