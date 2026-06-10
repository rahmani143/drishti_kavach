# import cv2
# import os

# # Import weapon detection modules
# from weapon_detection.main import initialize_weapon_model, detect_weapons

# # Import your face check function (adjusted to take frames)
# from face_recognition.app import run_security_check_on_frame

# def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
#     print("Initializing all models...")
    
#     # Initialize weapon model
#     weapon_detector = initialize_weapon_model(weapon_model_path)
    
#     # If your face recognition requires initializing weights or a database, do it here:
#     # face_database = initialize_face_database()

#     # Open the input video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     # Get video properties for output writing
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Setup Video Writer to save results
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     # Setup display windows
#     cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Drishti Kavach Integrated System', 800, 600)

#     print("\nStarting video processing. Press 'q' to exit early.")
    
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Finished processing all frames or video ended.")
#             break

#         frame_count += 1
#         print(f"Processing Frame: {frame_count}", end="\r")

#         # --- Pipeline Step A: Run Weapon Detection ---
#         # This function updates the frame with red bounding boxes internally
#         frame, weapon_data = detect_weapons(frame, weapon_detector)

#         # --- Pipeline Step B: Run Face Recognition & Risk Assessment ---
#         # Pass the current frame directly into your adapted security pipeline
#         face_detections = run_security_check_on_frame(frame)
        
#         # Draw face detection results on the same frame
#         for face in face_detections:
#             x1, y1, x2, y2 = face["bbox"]
#             name = face["name"]
#             risk = face["risk"]
            
#             # Choose border color based on risk assessment
#             color = (0, 0, 255) if "High" in str(risk) else (0, 255, 0)
            
#             # Draw green/red bounding boxes for faces
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, f"{name} ({risk})", 
#                         (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                         0.7, color, 2)

#         # Write the fully annotated frame (faces + weapons) to the output file
#         out.write(frame)

#         # Display the combined results in real time
#         cv2.imshow('Drishti Kavach Integrated System', frame)
        
#         # Break loop if user presses 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nProcessing interrupted by user.")
#             break

#     # Clean up resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"\nProcessing complete. Output saved to: {output_path}")

# if __name__ == "__main__":
#     # Paths configuration
#     VIDEO_INPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\try.mp4"
#     VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
#     WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"

#     run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)







# edit 2:
# with speeds











# import cv2
# import os

# # Import weapon detection modules
# from weapon_detection.main import initialize_weapon_model, detect_weapons

# # Import your face check function (adjusted to take frames)
# from face_recognition.app import run_security_check_on_frame

# def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
#     print("Initializing all models...")
    
#     # Initialize weapon model
#     weapon_detector = initialize_weapon_model(weapon_model_path)

#     # Open the input video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     # Get video properties for output writing
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Setup Video Writer to save results
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     # Setup display windows
#     cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Drishti Kavach Integrated System', 800, 600)

#     print("\nStarting video processing. Press 'q' to exit early.")
    
#     frame_count = 0

#     # Caches to store the latest data
#     last_weapon_data = []
#     last_face_detections = []

#     target_width = 960
#     target_height = 540
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
        
#         # Smooth out the processing spike by staggering the AI models
#         if frame_count % 4 == 1:
#             # Frame 1, 5, 9... Run ONLY Weapon Detection
#             frame, last_weapon_data = detect_weapons(frame, weapon_detector)
#         elif frame_count % 4 == 3:
#             # Frame 3, 7, 11... Run ONLY Face Recognition
#             last_face_detections = run_security_check_on_frame(frame)
#         else:
#             # Frames 2, 4, 6, 8, 10... Skip all AI inference entirely!
#             # Draw cached weapon boxes manually on skipped frames
#             for weapon in last_weapon_data:
#                 bbox = weapon.get("bbox")
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{weapon['class_name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Always draw face boxes (either fresh or cached) on every frame
#         for face in last_face_detections:
#             x1, y1, x2, y2 = face["bbox"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{face['name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         out.write(frame)
#         cv2.imshow('Drishti Kavach Integrated System', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         out.write(frame)
#         cv2.imshow('Drishti Kavach Integrated System', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         # Clean up resources
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#         print(f"\nProcessing complete. Output saved to: {output_path}")

# if __name__ == "__main__":
#     VIDEO_INPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\video_input\test\test2.mp4"
#     VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
#     WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"

#     run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)







# import cv2
# import os
# import sys

# # Ensure Python can see subfolders relative to this file
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # Import weapon detection modules
# from weapon_detection.main import initialize_weapon_model, detect_weapons

# # Import your face check function (adjusted to take frames)
# from face_recognition.app import run_security_check_on_frame

# def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
#     print("Initializing all models...")
    
#     # Initialize weapon model
#     weapon_detector = initialize_weapon_model(weapon_model_path)

#     # Open the input video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     # Setup targeted downscaled dimensions for faster performance
#     target_width = 960
#     target_height = 540
#     fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Fallback default check if PROP_FPS fails
#     video_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     if video_fps > 0:
#         fps = video_fps

#     # Setup Video Writer to match our downscaled target resolution
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

#     # Setup display windows
#     cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Drishti Kavach Integrated System', 800, 600)

#     print("\nStarting video processing. Press 'q' to exit early.")
    
#     frame_count = 0

#     # Caches to store the latest detection data on skipped frames
#     last_weapon_data = []
#     last_face_detections = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("\nFinished processing all frames or video ended.")
#             break

#         frame_count += 1
#         print(f"Processing Frame: {frame_count}", end="\r")

#         # FIX 1: Downscale the frame matrix immediately to match the VideoWriter
#         frame = cv2.resize(frame, (target_width, target_height))
        
#         # Smooth out processing spikes by staggering the AI models
#         if frame_count % 4 == 1:
#             # Frame 1, 5, 9... Run ONLY Weapon Detection
#             frame, last_weapon_data = detect_weapons(frame, weapon_detector)
#         elif frame_count % 4 == 3:
#             # Frame 3, 7, 11... Run ONLY Face Recognition
#             last_face_detections = run_security_check_on_frame(frame)
#         else:
#             # Frames 2, 4, 6, 8, 10... Skip all AI inference entirely!
#             # Draw cached weapon boxes manually on skipped frames to preserve continuity
#             for weapon in last_weapon_data:
#                 bbox = weapon.get("bbox")
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{weapon['class_name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Always draw face boxes (either fresh or cached) on every frame
#         for face in last_face_detections:
#             x1, y1, x2, y2 = face["bbox"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{face['name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Save frame to disk and push to preview layout window
#         out.write(frame)
#         cv2.imshow('Drishti Kavach Integrated System', frame)
        
#         # Break loop gracefully if user hits 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nProcessing interrupted by user.")
#             break

#     # FIX 3: Safe cleanup calls placed completely outside the processing loop block
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Processing complete. Output saved to: {output_path}")

# if __name__ == "__main__":
#     VIDEO_INPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\video_input\test\test2.mp4"
#     VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
#     WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"

#     run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)

















# edit 3:

# abnormally big













# import cv2
# import os
# import sys

# # Ensure Python can see subfolders relative to this file
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # Import weapon detection modules
# from weapon_detection.main import initialize_weapon_model, detect_weapons

# # Import your face check function (adjusted to take frames)
# from face_recognition.app import run_security_check_on_frame

# def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
#     print("Initializing all models...")
    
#     # Initialize weapon model
#     weapon_detector = initialize_weapon_model(weapon_model_path)

#     # Open the input video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     # --- DYNAMIC RESOLUTION DETECTOR ---
#     # Read properties directly from the source video metadata
#     native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     video_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     fps = video_fps if video_fps > 0 else 30

#     print(f"[INFO] Video detected with native resolution: {native_width}x{native_height} at {fps} FPS")

#     # Setup Video Writer to match the exact source resolution aspect ratio
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (native_width, native_height))

#     # Setup dynamic display windows
#     # cv2.WINDOW_AUTOSIZE forces the window to snap to the exact dimensions of the frame matrix
#     cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_AUTOSIZE)

#     print("\nStarting video processing. Press 'q' to exit early.")
    
#     frame_count = 0

#     # Caches to store the latest detection data on skipped frames
#     last_weapon_data = []
#     last_face_detections = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("\nFinished processing all frames or video ended.")
#             break

#         frame_count += 1
#         print(f"Processing Frame: {frame_count}", end="\r")
        
#         # Smooth out processing spikes by staggering the AI models
#         if frame_count % 4 == 1:
#             # Frame 1, 5, 9... Run ONLY Weapon Detection
#             frame, last_weapon_data = detect_weapons(frame, weapon_detector)
#         elif frame_count % 4 == 3:
#             # Frame 3, 7, 11... Run ONLY Face Recognition
#             last_face_detections = run_security_check_on_frame(frame)
#         else:
#             # Frames 2, 4, 6, 8, 10... Skip all AI inference entirely!
#             # Draw cached weapon boxes manually on skipped frames to preserve continuity
#             for weapon in last_weapon_data:
#                 bbox = weapon.get("bbox")
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{weapon['class_name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Always draw face boxes (either fresh or cached) on every frame
#         for face in last_face_detections:
#             x1, y1, x2, y2 = face["bbox"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{face['name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Save frame to disk and push to preview layout window natively
#         out.write(frame)
#         cv2.imshow('Drishti Kavach Integrated System', frame)
        
#         # Break loop gracefully if user hits 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nProcessing interrupted by user.")
#             break

#     # Safe cleanup calls placed completely outside the processing loop block
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Processing complete. Output saved to: {output_path}")

# if __name__ == "__main__":
#     VIDEO_INPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\video_input\test\test2.mp4"
#     VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
#     WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"

#     run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)















# edit 4:












# import cv2
# import os
# import sys

# # Ensure Python can see subfolders relative to this file
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # Import weapon detection modules
# from weapon_detection.main import initialize_weapon_model, detect_weapons

# # Import your face check function (adjusted to take frames)
# from face_recognition.app import run_security_check_on_frame

# def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
#     print("Initializing all models...")
    
#     # Initialize weapon model
#     weapon_detector = initialize_weapon_model(weapon_model_path)

#     # Open the input video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     # Read properties directly from the source video metadata
#     native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     video_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     fps = video_fps if video_fps > 0 else 30

#     print(f"[INFO] Video native resolution: {native_width}x{native_height}")

#     # Setup Video Writer to match the exact source resolution aspect ratio
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (native_width, native_height))

#     # --- DYNAMIC SCREEN-FIT MATHEMATICS ---
#     # Define the maximum safe dimensions your monitor window can occupy.
#     # Adjust these numbers if your monitor resolution is smaller or larger.
#     MAX_SCREEN_WIDTH = 1280
#     MAX_SCREEN_HEIGHT = 720

#     # Calculate the aspect ratio of the video (width / height)
#     aspect_ratio = native_width / native_height

#     # Default window sizes match the video
#     display_width = native_width
#     display_height = native_height

#     # If the video is too tall for your screen bounds, scale by height
#     if display_height > MAX_SCREEN_HEIGHT:
#         display_height = MAX_SCREEN_HEIGHT
#         display_width = int(display_height * aspect_ratio)

#     # If the video is still too wide for your screen bounds, scale by width instead
#     if display_width > MAX_SCREEN_WIDTH:
#         display_width = MAX_SCREEN_WIDTH
#         display_height = int(display_width / aspect_ratio)

#     print(f"[INFO] Adjusting window frame scale bounds to fit screen: {display_width}x{display_height}")

#     # Setup the display windows
#     # WINDOW_NORMAL allows the frame dimensions to scale away from 100% native resolution size
#     # Find this section in your script:
#     # cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('Drishti Kavach Integrated System', display_width, display_height)

#     # AND REPLACE IT WITH THIS:
#     cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
    
#     # Force the window manager to preserve the video's aspect ratio when maximized
#     cv2.setWindowProperty('Drishti Kavach Integrated System', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    
#     # Set the initial safe size that fits your screen
#     cv2.resizeWindow('Drishti Kavach Integrated System', display_width, display_height)

#     print("\nStarting video processing. Press 'q' to exit early.")
    
#     frame_count = 0

#     # Caches to store the latest detection data on skipped frames
#     last_weapon_data = []
#     last_face_detections = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("\nFinished processing all frames or video ended.")
#             break

#         frame_count += 1
#         print(f"Processing Frame: {frame_count}", end="\r")
        
#         # Smooth out processing spikes by staggering the AI models
#         # Smooth out processing spikes by staggering the AI models
#         if frame_count % 4 == 1:
#             # Create a clean copy so weapon box modifications do not distort the base image array
#             clean_frame_copy = frame.copy()
#             # Pass the copy to the weapon detector instead of the raw original frame
#             _, last_weapon_data = detect_weapons(clean_frame_copy, weapon_detector)
            
#         elif frame_count % 4 == 3:
#             # Frame 3 now receives a completely clean, un-annotated video matrix
#             last_face_detections = run_security_check_on_frame(frame)
#         else:
#             # Intermediate frames: Skip AI inference entirely.
#             # Draw cached weapon boxes manually on skipped frames to preserve continuity
#             for weapon in last_weapon_data:
#                 bbox = weapon.get("bbox")
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{weapon['class_name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Always draw face boxes (either fresh or cached) on every frame
#         for face in last_face_detections:
#             x1, y1, x2, y2 = face["bbox"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{face['name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Save frame to disk at original full resolution quality
#         out.write(frame)
        
#         # Push to the scaled preview layout window (OpenCV handles the compression automatically)
#         cv2.imshow('Drishti Kavach Integrated System', frame)
        
#         # Break loop gracefully if user hits 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nProcessing interrupted by user.")
#             break

#     # Safe cleanup calls placed completely outside the processing loop block
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Processing complete. Output saved to: {output_path}")

# if __name__ == "__main__":
#     VIDEO_INPUT = r"C:\Users\bss10\Downloads\videoplayback.mp4"
#     VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
#     WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"

#     run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)










# edit 5:












# import cv2
# import os
# import sys
# import pandas as pd

# # Ensure Python can see subfolders relative to this file
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # Import weapon detection modules
# from weapon_detection.main import initialize_weapon_model, detect_weapons
# # Import your face check function
# from face_recognition.app import run_security_check_on_frame

# # Path to your synthetic crime database CSV
# CSV_DB_PATH = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\crime_database.csv"

# def load_crime_database():
#     """Loads the CSV database into memory for fast lookup."""
#     if os.path.exists(CSV_DB_PATH):
#         print(f"[INFO] Loading crime database from {CSV_DB_PATH}...")
#         df = pd.read_csv(CSV_DB_PATH)
#         # Convert identity_name to index for direct O(1) dictionary lookup
#         return df.set_index("identity_name").to_dict(orient="index")
#     else:
#         print(f"[WARNING] Crime database CSV not found at {CSV_DB_PATH}. Alerting will be disabled.")
#         return {}

# def check_and_alert_identity(name, crime_db, alerted_identities):
#     """Checks the database for the matched identity and prints a structured alert."""
#     if not crime_db or name == "Unknown":
#         return

#     # If we have already triggered an alert for this person in this video run, skip log spam
#     if name in alerted_identities:
#         return

#     if name in crime_db:
#         profile = crime_db[name]
#         has_warrant = int(profile.get("has_active_warrant", 0))
#         prior_arrests = int(profile.get("prior_arrests", 0))
#         is_dangerous = int(profile.get("is_dangerous", 0))
#         months_since = profile.get("months_since_last_incident", "Unknown")

#         print("\n" + "="*60)
#         print(f"🚨 [MATCH DETECTED] Identity: {name}")
#         print(f"   - Prior Arrests: {prior_arrests}")
#         print(f"   - Months Since Last Incident: {months_since}")
#         print(f"   - Dangerous Flag: {'YES' if is_dangerous == 1 else 'NO'}")
#         print("-" * 60)
        
#         if has_warrant == 1:
#             print(f"🚨 [!!! ARREST REQUIRED !!!] {name} has an ACTIVE WARRANT. Detain immediately.")
#         else:
#             print(f"ℹ️ [MONITOR ONLY] {name} is recognized but has no active warrants.")
#         print("="*60 + "\n")
        
#         # Add to set so it only logs once
#         alerted_identities.add(name)

# def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
#     print("Initializing all models...")
    
#     # Load crime tracking data
#     crime_db = load_crime_database()
#     alerted_identities = set()  # Tracks who we already printed alerts for

#     # Initialize weapon model
#     weapon_detector = initialize_weapon_model(weapon_model_path)

#     # Open the input video file
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     # Read properties directly from the source video metadata
#     native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     video_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     fps = video_fps if video_fps > 0 else 30

#     print(f"[INFO] Video native resolution: {native_width}x{native_height}")

#     # Setup Video Writer to match the exact source resolution aspect ratio
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (native_width, native_height))

#     # --- DYNAMIC SCREEN-FIT MATHEMATICS ---
#     MAX_SCREEN_WIDTH = 1280
#     MAX_SCREEN_HEIGHT = 720
#     aspect_ratio = native_width / native_height

#     display_width = native_width
#     display_height = native_height

#     if display_height > MAX_SCREEN_HEIGHT:
#         display_height = MAX_SCREEN_HEIGHT
#         display_width = int(display_height * aspect_ratio)

#     if display_width > MAX_SCREEN_WIDTH:
#         display_width = MAX_SCREEN_WIDTH
#         display_height = int(display_width / aspect_ratio)

#     print(f"[INFO] Adjusting window frame scale bounds to fit screen: {display_width}x{display_height}")

#     cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty('Drishti Kavach Integrated System', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
#     cv2.resizeWindow('Drishti Kavach Integrated System', display_width, display_height)

#     print("\nStarting video processing. Press 'q' to exit early.\n")
    
#     frame_count = 0
#     last_weapon_data = []
#     last_face_detections = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("\nFinished processing all frames or video ended.")
#             break

#         frame_count += 1
        
#         # Smooth out processing spikes by staggering the AI models
#         if frame_count % 4 == 1:
#             clean_frame_copy = frame.copy()
#             _, last_weapon_data = detect_weapons(clean_frame_copy, weapon_detector)
            
#         elif frame_count % 4 == 3:
#             last_face_detections = run_security_check_on_frame(frame)
            
#             # Check face match names against database immediately when updated
#             for face in last_face_detections:
#                 check_and_alert_identity(face.get("name"), crime_db, alerted_identities)
#         else:
#             # Intermediate frames: Draw cached weapon boxes manually
#             for weapon in last_weapon_data:
#                 bbox = weapon.get("bbox")
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{weapon['class_name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Always draw face boxes (either fresh or cached) on every frame
#         for face in last_face_detections:
#             x1, y1, x2, y2 = face["bbox"]
#             name = face["name"]
            
#             # Dynamic coloring: Red if they have an active warrant in our system
#             is_wanted = crime_db.get(name, {}).get("has_active_warrant", 0) == 1
#             box_color = (0, 0, 255) if is_wanted else (0, 255, 0)
#             label = f"{name} [WANTED]" if is_wanted else name

#             cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

#         # Save frame to disk at original full resolution quality
#         out.write(frame)
        
#         # Push to display layout
#         cv2.imshow('Drishti Kavach Integrated System', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nProcessing interrupted by user.")
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Processing complete. Output saved to: {output_path}")

# if __name__ == "__main__":
#     VIDEO_INPUT = r"C:\Users\bss10\Downloads\videoplayback.mp4"
#     VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
#     WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"

#     run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)











# import cv2
# import os
# import sys
# import pandas as pd

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from weapon_detection.main import initialize_weapon_model, detect_weapons
# from face_recognition.app import run_security_check_on_frame

# # Path to your synthetic crime database CSV
# CSV_DB_PATH = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\crime_database.csv"

# def load_crime_database():
#     """Loads the CSV database into memory for rapid dictionary lookup."""
#     if os.path.exists(CSV_DB_PATH):
#         print(f"[INFO] Loading crime database records from {CSV_DB_PATH}...")
#         df = pd.read_csv(CSV_DB_PATH)
#         return df.set_index("identity_name").to_dict(orient="index")
#     print(f"[WARNING] Crime database CSV file not found at {CSV_DB_PATH}.")
#     return {}

# def process_security_alerts(face_data, crime_db, alerted_identities):
#     """Evaluates detections against the CSV profile and runs a single alert block if required."""
#     for face in face_data:
#         name = face.get("matched_name")
        
#         # Only run alerts if a known name is confirmed and has not been logged yet
#         if name and name in crime_db and name not in alerted_identities:
#             profile = crime_db[name]
#             warrant = int(profile.get("has_active_warrant", 0))
#             arrests = int(profile.get("prior_arrests", 0))
#             weapons = int(profile.get("weapon_charges_count", 0))
#             months = profile.get("months_since_last_incident", 0)
#             dangerous = int(profile.get("is_dangerous", 0))

#             print("\n" + "="*60)
#             print(f"[MATCH CONFIRMED] Identity Located: {name}")
#             print(f"Criminal History Profile:")
#             print(f"  - Prior Arrests Count: {arrests}")
#             print(f"  - Weapon Charges Count: {weapons}")
#             print(f"  - Months Since Last Incident: {months}")
#             print(f"  - Classified Dangerous: {'YES' if dangerous == 1 else 'NO'}")
#             print("-" * 60)
            
#             if warrant == 1:
#                 print(f"[ARREST MANDATORY] Subject {name} has an ACTIVE WARRANT. Take immediate custody.")
#             else:
#                 print(f"[MONITOR STATUS] Subject {name} identified. Clear of active warrants.")
#             print("="*60 + "\n")
            
#             # Lock out further prints for this person during this execution run
#             alerted_identities.add(name)

# def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
#     print("Initializing components...")
#     crime_db = load_crime_database()
#     alerted_identities = set()

#     weapon_detector = initialize_weapon_model(weapon_model_path)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Unable to open video source {video_path}")
#         return

#     native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     video_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     fps = video_fps if video_fps > 0 else 30

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (native_width, native_height))

#     MAX_SCREEN_WIDTH = 1280
#     MAX_SCREEN_HEIGHT = 720
#     aspect_ratio = native_width / native_height
#     display_width, display_height = native_width, native_height

#     if display_height > MAX_SCREEN_HEIGHT:
#         display_height = MAX_SCREEN_HEIGHT
#         display_width = int(display_height * aspect_ratio)
#     if display_width > MAX_SCREEN_WIDTH:
#         display_width = MAX_SCREEN_WIDTH
#         display_height = int(display_width / aspect_ratio)

#     cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty('Drishti Kavach Integrated System', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
#     cv2.resizeWindow('Drishti Kavach Integrated System', display_width, display_height)

#     print("\nProcessing streaming input. Press 'q' to terminate.\n")
#     frame_count = 0
#     last_weapon_data = []
#     last_face_detections = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         if frame_count % 4 == 1:
#             clean_frame_copy = frame.copy()
#             _, last_weapon_data = detect_weapons(clean_frame_copy, weapon_detector)
#         elif frame_count % 4 == 3:
#             last_face_detections = run_security_check_on_frame(frame)
#             # Evaluate against CSV data and report if matched
#             process_security_alerts(last_face_detections, crime_db, alerted_identities)
#         else:
#             for weapon in last_weapon_data:
#                 bbox = weapon.get("bbox")
#                 if bbox:
#                     x1, y1, x2, y2 = bbox
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                     cv2.putText(frame, f"{weapon['class_name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Draw box parameters onto frame
#         for face in last_face_detections:
#             x1, y1, x2, y2 = face["bbox"]
#             matched_name = face["matched_name"]
#             similarity = face["similarity"]
#             best_match = face["best_match_name"]

#             if matched_name:
#                 is_wanted = crime_db.get(matched_name, {}).get("has_active_warrant", 0) == 1
#                 box_color = (0, 0, 255) if is_wanted else (0, 255, 0)
#                 label = f"{matched_name} [WANTED]" if is_wanted else f"{matched_name} ({similarity:.2f})"
#             else:
#                 box_color = (0, 255, 255)
#                 label = f"Unknown (Closest: {best_match} {similarity:.2f})"

#             cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

#         out.write(frame)
#         cv2.imshow('Drishti Kavach Integrated System', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("\nProcessing manually stopped.")
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Processing complete. Video file saved to: {output_path}")

# if __name__ == "__main__":
#     VIDEO_INPUT = r"C:\Users\bss10\Downloads\videoplayback.mp4"
#     VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
#     WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"
#     run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)









# edit :











import cv2
import os
import sys
import pandas as pd
import threading
import winsound


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from weapon_detection.main import initialize_weapon_model, detect_weapons
from face_recognition.app import run_security_check_on_frame

# Path to your synthetic crime database CSV
CSV_DB_PATH = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\crime_database.csv"

def load_crime_database():
    """Loads the CSV database into memory for rapid dictionary lookup."""
    if os.path.exists(CSV_DB_PATH):
        print(f"[INFO] Loading crime database records from {CSV_DB_PATH}...")
        df = pd.read_csv(CSV_DB_PATH)
        return df.set_index("identity_name").to_dict(orient="index")
    print(f"[WARNING] Crime database CSV file not found at {CSV_DB_PATH}.")
    return {}

def trigger_audio_alarm():
    """Plays a sharp, three-beep hardware audio alarm in a separate background thread."""
    def beep_sequence():
        for _ in range(3):
            winsound.Beep(2500, 150)  # Frequency: 2500Hz, Duration: 150ms
    threading.Thread(target=beep_sequence, daemon=True).start()

def process_security_alerts(face_data, crime_db, alerted_identities):
    """Evaluates detections against the CSV profile and runs a single alert block if required."""
    for face in face_data:
        matched_name = face.get("matched_name")
        best_match = face.get("best_match_name")
        similarity = face.get("similarity", 0.0)
        face_detected = face.get("face_detected", False)
        
        # --- NEW CODE FOR UNKNOWN/LOW CONFIDENCE DETECTION AUDIO ---
        # If a real face is detected but cannot be verified with high confidence,
        # trigger the audio alarm immediately without checking the CSV database.
        if matched_name is None and face_detected:
            # Create a unique identifier string to prevent repeating the audio loop for the same clip block
            unknown_id = f"unknown_closest_{best_match}"
            if unknown_id not in alerted_identities:
                print(f"[SECURITY ALERT] Unknown person detected. Closest match: {best_match}")
                trigger_audio_alarm()
                alerted_identities.add(unknown_id)
            continue
        # -----------------------------------------------------------

        # If confidence is high, check standard database rules
        if matched_name is None:
            target_name = best_match
            is_unverified = True
        else:
            target_name = matched_name
            is_unverified = False

        if not target_name:
            continue

        # Trigger alert if the target name exists in the database and hasn't alerted yet
        if target_name in crime_db and target_name not in alerted_identities:
            profile = crime_db[target_name]
            warrant = int(profile.get("has_active_warrant", 0))
            arrests = int(profile.get("prior_arrests", 0))
            weapons = int(profile.get("weapon_charges_count", 0))
            dangerous = int(profile.get("is_dangerous", 0))

            # Only sound the alarm for an unverified match if they are flagged as dangerous or have a warrant
            if is_unverified and warrant == 0 and dangerous == 0:
                print(f"[INFO] Low confidence match for {target_name} ({similarity:.2f}). No critical threat factors. Skipping alarm.")
                continue

            print("\n" + "="*60)
            if is_unverified:
                print(f"[CRITICAL ALERT] Unverified Match suspect identified as: {target_name} (Confidence: {similarity:.2f})")
            else:
                print(f"[MATCH CONFIRMED] Identity Located: {target_name}")
                
            print(f"Criminal History Profile for {target_name}:")
            print(f"  - Prior Arrests Count: {arrests}")
            print(f"  - Weapon Charges Count: {weapons}")
            print(f"  - Classified Dangerous: {'YES' if dangerous == 1 else 'NO'}")
            print("-" * 60)
            
            if warrant == 1:
                print(f"[ARREST MANDATORY] Subject {target_name} has an ACTIVE WARRANT. Take immediate custody.")
            
            print("="*60 + "\n")
            
            # Sound the physical hardware alarm
            trigger_audio_alarm()
            
            # Prevent repetitive spamming for this profile
            alerted_identities.add(target_name)

def run_combined_video_pipeline(video_path, output_path, weapon_model_path):
    print("Initializing components...")
    crime_db = load_crime_database()
    alerted_identities = set()

    weapon_detector = initialize_weapon_model(weapon_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return

    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = video_fps if video_fps > 0 else 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (native_width, native_height))

    MAX_SCREEN_WIDTH = 1280
    MAX_SCREEN_HEIGHT = 720
    aspect_ratio = native_width / native_height
    display_width, display_height = native_width, native_height

    if display_height > MAX_SCREEN_HEIGHT:
        display_height = MAX_SCREEN_HEIGHT
        display_width = int(display_height * aspect_ratio)
    if display_width > MAX_SCREEN_WIDTH:
        display_width = MAX_SCREEN_WIDTH
        display_height = int(display_width / aspect_ratio)

    cv2.namedWindow('Drishti Kavach Integrated System', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Drishti Kavach Integrated System', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Drishti Kavach Integrated System', display_width, display_height)

    print("\nProcessing streaming input. Press 'q' to terminate.\n")
    frame_count = 0
    last_weapon_data = []
    last_face_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Run model inferences on alternating frames to preserve processing speed
        if frame_count % 4 == 1:
            clean_frame_copy = frame.copy()
            _, last_weapon_data = detect_weapons(clean_frame_copy, weapon_detector)
        elif frame_count % 4 == 3:
            last_face_detections = run_security_check_on_frame(frame)
            process_security_alerts(last_face_detections, crime_db, alerted_identities)

        # 2. Draw weapons on EVERY frame using the cached coordinates
        for weapon in last_weapon_data:
            bbox = weapon.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{weapon['class_name']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3. Draw faces on EVERY frame using the cached coordinates
        for face in last_face_detections:
            x1, y1, x2, y2 = face["bbox"]
            clean_name = face.get("matched_name") # Read exact key from app.py
            similarity = face.get("similarity", 0.0)
            best_match = face.get("best_match_name", "Unknown")

            if clean_name is not None:
                # Name verified in dataset above threshold limits
                is_wanted = crime_db.get(clean_name, {}).get("has_active_warrant", 0) == 1
                box_color = (0, 0, 255) if is_wanted else (0, 255, 0) # Red for warrants, Green for verified
                label = f"{clean_name} [WANTED]" if is_wanted else f"{clean_name} ({similarity:.2f})"
            else:
                # Low confidence or completely unknown
                box_color = (0, 255, 255) # Yellow box for unverified/unknown
                label = f"Unknown (Closest: {best_match})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        out.write(frame)
        cv2.imshow('Drishti Kavach Integrated System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcessing manually stopped.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Video file saved to: {output_path}")

if __name__ == "__main__":
    VIDEO_INPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\video_input\test\15-43-30.mp4"
    VIDEO_OUTPUT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\processed_video.mp4"
    WEAPON_MODEL = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"
    run_combined_video_pipeline(VIDEO_INPUT, VIDEO_OUTPUT, WEAPON_MODEL)