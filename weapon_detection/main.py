import cv2
import os
from ultralytics import YOLO

def initialize_weapon_model(yolo_model_path):
    print("Loading YOLOv8 Weapon Detection Model...")
    model = YOLO(yolo_model_path)
    return model

def detect_weapons(frame, model):
    # conf=0.25 is a standard starting confidence threshold for YOLO models
    results = model(frame, conf=0.12, verbose=False)[0]
    
    detected_weapons = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        detected_weapons.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "class_name": class_name
        })
        
        # Draw red bounding boxes for weapons
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{class_name} ({confidence:.2f})", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 0, 255), 2)

    return frame, detected_weapons

if __name__ == "__main__":
    # Specify the path to your test images and your trained weapon model
    folder_path = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\crime.v10i.yolov8\test\images"
    model_path = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\weapon_detection\best.pt"
    
    detector = initialize_weapon_model(model_path)
    
    cv2.namedWindow('Weapon Detection Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Weapon Detection Output', 800, 600) 
    
    for filename in os.listdir(folder_path):
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                print(f"\nProcessing file: {filename}")
                processed_image, weapon_data = detect_weapons(image, detector)
                
                for i, data in enumerate(weapon_data):
                    print(f"Detection {i+1}: {data['class_name']} | Confidence: {data['confidence']:.2f}")
                
                cv2.imshow('Weapon Detection Output', processed_image)
                
                key = cv2.waitKey(0) & 0xFF
                if key == 27 or key == ord('q'):
                    print("Quitting process...")
                    break
            else:
                print(f"Error: Could not load the image {filename}.")
                
    cv2.destroyAllWindows()