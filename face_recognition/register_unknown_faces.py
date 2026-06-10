# import os
# import sys
# import subprocess
# import cv2

# def register_new_identities():
#     # 1. Define your path configurations
#     UNKNOWN_DIR = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\unknown_faces"
#     DATASET_BASE_DIR = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces"
#     BUILD_DB_SCRIPT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\build_database.py"

#     if not os.path.exists(UNKNOWN_DIR):
#         print(f"Error: Directory does not exist -> {UNKNOWN_DIR}")
#         return

#     # Get all image files from the unknown directory
#     unknown_images = [f for f in os.listdir(UNKNOWN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     if not unknown_images:
#         print("No unknown face snapshots found to register.")
#         return

#     print(f"Found {len(unknown_images)} unknown snapshots. Starting interactive registration...\n")
#     cv2.namedWindow("Verify Unknown Face", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Verify Unknown Face", 400, 400)

#     registered_any = False

#     for filename in unknown_images:
#         img_path = os.path.join(UNKNOWN_DIR, filename)
#         img = cv2.imread(img_path)

#         if img is None:
#             continue

#         # Display the face image so you know who you are naming
#         cv2.imshow("Verify Unknown Face", img)
#         cv2.waitKey(500)  # Brief pause to render the window clearly

#         print(f"Showing face from file: {filename}")
#         name_input = input("Enter the name for this person (or press Enter to SKIP/DELETE): ").strip()

#         if name_input:
#             # Clean up the input string to make a valid directory name (lowercase with underscores)
#             folder_name = name_input.lower().replace(" ", "_")
#             target_folder = os.path.join(DATASET_BASE_DIR, folder_name)
            
#             # Create the directory if it does not exist yet
#             os.makedirs(target_folder, exist_ok=True)

#             # Define the final path inside your training dataset folder
#             destination_path = os.path.join(target_folder, f"{folder_name}_{int(os.path.getmtime(img_path))}.jpg")
            
#             # Save the image to the new location
#             cv2.imwrite(destination_path, img)
#             print(f"[SUCCESS] Moved and saved image to: {destination_path}")
#             registered_any = True
#         else:
#             print("[INFO] Skipped naming this person.")

#         # Clean up the processed file from the raw snapshots folder so you don't look at it next time
#         cv2.destroyWindow("Verify Unknown Face")
#         try:
#             os.remove(img_path)
#         except Exception as e:
#             print(f"Warning: Could not remove temporary snapshot file: {e}")

#     cv2.destroyAllWindows()

#     # 2. Automatically rebuild the compilation vector file if changes were made
#     if registered_any:
#         print("\n" + "="*50)
#         print("New faces added. Re-running database compilation script...")
#         print("="*50 + "\n")
        
#         try:
#             # Run 'uv run' as a subprocess tool inside your environment pipeline
#             result = subprocess.run(["uv", "run", BUILD_DB_SCRIPT], capture_output=False, text=True, check=True)
#             print("\n[SYSTEM NOTICE] Database compilation completed successfully!")
#         except subprocess.CalledProcessError as err:
#             print(f"\n[ERROR] Failed to run database compilation automatically: {err}")
#     else:
#         print("\nNo new database entries were processed. Compilation skipped.")

# if __name__ == "__main__":
#     register_new_identities()

















# edit 2:














import os
import sys
import subprocess
import cv2

def register_new_identities():
    # 1. Path Configurations
    UNKNOWN_DIR = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\unknown_faces"
    DATASET_BASE_DIR = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces"
    BUILD_DB_SCRIPT = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\main.py"     

    # Initialize OpenCV's built-in face detector
    # This XML file is included by default with every OpenCV installation
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.getStructuringElement if not os.path.exists(cascade_path) else cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade face detector.")
        return

    if not os.path.exists(UNKNOWN_DIR):
        print(f"Error: Directory does not exist -> {UNKNOWN_DIR}")
        return

    unknown_images = [f for f in os.listdir(UNKNOWN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not unknown_images:
        print("No unknown face snapshots found to register.")
        return

    print(f"Found {len(unknown_images)} snapshots. Starting interactive registration...\n")
    registered_any = False

    for filename in unknown_images:
        img_path = os.path.join(UNKNOWN_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Convert to grayscale for the face detector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect the exact face boundaries inside the snapshot
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"[SKIP] No clear facial features detected in file: {filename}. Skipping to avoid database corruption.")
            try:
                os.remove(img_path)
            except Exception:
                pass
            continue

        # Extract the coordinates of the largest detected face area
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Crop strictly to the face region with a small padding margin (10%)
        pad_w = int(w * 0.1)
        pad_h = int(h * 0.1)
        img_h, img_w, _ = img.shape
        
        y1 = max(0, y - pad_h)
        y2 = min(img_h, y + h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(img_w, x + w + pad_w)
        
        cropped_face = img[y1:y2, x1:x2]

        # Render preview windows
        cv2.namedWindow("Full Image Context", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Full Image Context", 400, 400)
        cv2.namedWindow("Face Crop Output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Crop Output", 200, 200)

        # Draw a temporary box on the preview image to show what is being cropped
        preview_img = img.copy()
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Full Image Context", preview_img)
        cv2.imshow("Face Crop Output", cropped_face)
        cv2.waitKey(500)

        print(f"\nProcessing file: {filename}")
        name_input = input("Enter the name for this person (or press Enter to SKIP/DELETE): ").strip()

        if name_input:
            folder_name = name_input.lower().replace(" ", "_")
            target_folder = os.path.join(DATASET_BASE_DIR, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            # Define destination path
            destination_path = os.path.join(target_folder, f"{folder_name}_{int(os.path.getmtime(img_path))}.jpg")
            
            # Save the CLEAN CROPPED FACE matrix instead of the raw original image
            cv2.imwrite(destination_path, cropped_face)
            print(f"[SUCCESS] Saved clean face crop to: {destination_path}")
            registered_any = True
        else:
            print("[INFO] Skipped naming this person.")

        # Clean up viewports and delete processed file
        cv2.destroyAllWindows()
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")

    # 2. Rebuild the vector database completely if changes were made
    if registered_any:
        print("\n" + "="*50)
        print("Appending new face vector directly to face_database.npz...")
        print("="*50 + "\n")
        
        # --- ADD THESE LINES TO FIX THE PATHS ---
        import sys
        # Find the parent folder path dynamically
        PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if PARENT_DIR not in sys.path:
            sys.path.insert(0, PARENT_DIR)
        # ----------------------------------------

        # Now this import will find 'face_recognition' without throwing an error
        from face_recognition.main import append_single_person_to_database
        
        success = append_single_person_to_database(folder_name, target_folder)
        if success:
            print("[SYSTEM NOTICE] Database update complete!")
        else:
            print("[ERROR] Could not append data to database.")

if __name__ == "__main__":
    register_new_identities()