import os
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request

# 1. Download the pre-trained ArcFace model weights natively using Python
MODEL_PATH = "arcface.onnx"
if not os.path.exists(MODEL_PATH):
    print("Downloading ArcFace weights... This might take a minute.")
    url = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

# 2. Initialize the ArcFace ONNX Session
print("Loading ArcFace model...")
arcface_sess = ort.InferenceSession(MODEL_PATH)
arc_input_name = arcface_sess.get_inputs()[0].name
arc_output_name = arcface_sess.get_outputs()[0].name

# 3. Initialize the OpenCV Face Detector
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# ... keep your get_identity_vector function the same below this ...

def get_identity_vector(image_input):
    # Check if the input is a string (file path) or a NumPy array (video frame)
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        # It's already a numpy array/frame from the video pipeline, use it directly
        img = image_input

    if img is None:
        print("Error: Image frame could not be loaded.")
        return None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect the face bounding box
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    # Crop the face or default to a center crop if detection fails
    if len(faces) == 0:
        h, w, _ = img_rgb.shape
        side = min(w, h)
        face_crop = img_rgb[(h-side)//2 : (h+side)//2, (w-side)//2 : (w+side)//2]
    else:
        x, y, w, h = faces[0]
        face_crop = img_rgb[y:y+h, x:x+w]
        
    # Resize exactly to 112x112 pixels for ArcFace
    resized = cv2.resize(face_crop, (112, 112))
    
    # Normalize pixel values
    normalized = (resized.astype(np.float32) - 127.5) / 128.0
    input_tensor = normalized[np.newaxis, ...]
    
    # Generate the 512-dimensional identity vector
    embedding = arcface_sess.run([arc_output_name], {arc_input_name: input_tensor})[0][0]
    
    # Normalize the vector length to 1.0 for cosine similarity matching
    final_vector = embedding / np.linalg.norm(embedding)
    
    return final_vector

DB_PATH = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\face_database.npz"
SECURITY_THRESHOLD = 0.65  # Minimum cosine similarity to confirm a match

def build_database(targets_dir="targets"):
    """
    Scans the targets folder, generates vectors for each person, 
    and saves them to a local database file.
    """
    if not os.path.exists(targets_dir):
        print(f"Error: Targets directory '{targets_dir}' does not exist.")
        return
        
    database = {}
    print("\n--- Building Face Database ---")
    
    for person_name in os.listdir(targets_dir):
        person_path = os.path.join(targets_dir, person_name)
        if not os.path.isdir(person_path):
            continue
            
        vectors = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            vector = get_identity_vector(img_path)
            if vector is not None:
                vectors.append(vector)
                
        if vectors:
            # Average the vectors if there are multiple photos, then normalize
            mean_vector = np.mean(vectors, axis=0)
            database[person_name] = mean_vector / np.linalg.norm(mean_vector)
            print(f"Registered: {person_name}")
            
    # Save the database to disk
    np.savez(DB_PATH, **database)
    print(f"Database successfully saved to {DB_PATH}\n")

def identify_face(image_path):
    """
    Compares a target face against the database. 
    Returns:
        - matched_name (str or None): The name if above SECURITY_THRESHOLD, else None.
        - max_similarity (float): The actual calculated similarity score.
        - best_match (str): The name of the closest person found in the database.
    """
    if not os.path.exists(DB_PATH):
        print("Error: Database file not found. Run build_database() first.")
        return None, 0.0, "Unknown"
        
    # Load the database
    with np.load(DB_PATH) as data:
        database = {key: data[key] for key in data.files}
        
    test_vector = get_identity_vector(image_path)
    if test_vector is None:
        print("Could not process the query image.")
        return None, 0.0, "Unknown"
        
    best_match = "Unknown"
    max_similarity = -1.0
    
    # Linear scan matching
    for identity_name, reference_vector in database.items():
        similarity = np.dot(test_vector, reference_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = identity_name
            
    # Apply open-set threshold check for terminal logs
    if max_similarity >= SECURITY_THRESHOLD:
        print(f"[MATCH FOUND] Identity: {best_match} | Confidence: {max_similarity:.4f}")
        return best_match, float(max_similarity), best_match
    else:
        # print(f"[UNKNOWN] Identity not recognized. Highest similarity: {max_similarity:.4f} with {best_match}")
        # Return None for strict match, but still return the similarity and best match name
        return None, float(max_similarity), best_match

def append_single_person_to_database(person_name, person_folder_path):
    """
    Appends a single new person to the existing face_database.npz 
    without re-processing the entire dataset.
    """
    # 1. Load the existing database dictionary into memory
    if os.path.exists(DB_PATH):
        print(f"Loading existing database from {DB_PATH}...")
        with np.load(DB_PATH) as data:
            database = {key: data[key] for key in data.files}
    else:
        print("Database file does not exist yet. Creating a new one.")
        database = {}

    # 2. Process only the images inside the new person's folder
    if not os.path.exists(person_folder_path):
        print(f"Error: Folder path {person_folder_path} does not exist.")
        return False

    vectors = []
    print(f"Generating identity vector for: {person_name}...")
    for img_name in os.listdir(person_folder_path):
        img_path = os.path.join(person_folder_path, img_name)
        vector = get_identity_vector(img_path)
        if vector is not None:
            vectors.append(vector)

    if not vectors:
        print(f"Error: Could not extract any valid face vectors for {person_name}.")
        return False

    # 3. Calculate mean vector and normalize
    mean_vector = np.mean(vectors, axis=0)
    normalized_vector = mean_vector / np.linalg.norm(mean_vector)

    # 4. Add or update the dictionary entry
    database[person_name] = normalized_vector

    # 5. Save the updated dictionary back to the same NPZ file path
    np.savez(DB_PATH, **database)
    print(f"[SUCCESS] {person_name} has been added directly to the database file.")
    return True

if __name__ == "__main__":
    # Define the absolute path to your Indian actors dataset
    DATASET_DIR = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces"
    
    # 1. Build the database using your actors folder (Run this once to generate face_database.npz)
    build_database(targets_dir=DATASET_DIR)
    
    # 2. Test identification with a specific image path to verify
    # test_image = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\face_recognition\dataset\actors_dataset\Indian_actors_faces\abhay_deol\3fe4d478d0.jpg"
    # identify_face(test_image)

print("Face Recognition Module Ready.")