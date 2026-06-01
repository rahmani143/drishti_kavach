# import cv2
# import easyocr
# from ultralytics import YOLO

# def initialize_anpr_models(yolo_model_path='yolov8n_plates.pt'):
#     """
#     Initializes both the YOLOv8 detector and the EasyOCR reader.
#     """
#     print("Loading YOLOv8 License Plate Model...")
#     # Load your custom fine-tuned YOLOv8 model for plates
#     plate_detector = YOLO(yolo_model_path)
    
#     print("Loading EasyOCR Reader...")
#     # Initialize EasyOCR for English text
#     # Setting gpu=False ensures it runs on CPU, ideal for a Raspberry Pi setup
#     ocr_reader = easyocr.Reader(['en'], gpu=False) 
    
#     return plate_detector, ocr_reader

# def read_license_plate(frame, plate_detector, ocr_reader):
#     """
#     Detects a license plate in a frame and extracts the text.
#     """
#     # 1. Run YOLOv8 inference to find the license plate
#     results = plate_detector(frame, conf=0.5)[0]
    
#     detected_plates = []

#     # 2. Iterate over the detections
#     for box in results.boxes:
#         # Extract the bounding box coordinates
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         confidence = float(box.conf[0])
        
#         # 3. Crop the detected license plate from the original frame
#         plate_crop = frame[y1:y2, x1:x2]
        
#         # Optional: Convert the crop to grayscale to improve OCR accuracy
#         gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
#         # 4. Pass the cropped image to EasyOCR to read the text
#         ocr_results = ocr_reader.readtext(gray_plate)
        
#         plate_text = ""
#         text_confidence = 0.0
        
#         # 5. Parse the OCR results (EasyOCR returns a list of tuples: [bbox, text, confidence])
#         if ocr_results:
#             # Taking the highest confidence text found in the crop
#             # You can also concatenate strings if the plate is read in multiple chunks
#             best_read = max(ocr_results, key=lambda x: x[2])
#             plate_text = best_read[1].upper() # Convert to uppercase for uniformity
#             text_confidence = best_read[2]
            
#         # Store the structured data
#         detected_plates.append({
#             "bbox": (x1, y1, x2, y2),
#             "detection_confidence": confidence,
#             "plate_text": plate_text,
#             "ocr_confidence": text_confidence
#         })
        
#         # --- VISUALIZATION (Optional) ---
#         # Draw the bounding box and text on the frame for visual confirmation
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{plate_text} ({text_confidence:.2f})", 
#                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.9, (0, 255, 0), 2)

#     return frame, detected_plates

# # --- EXECUTION ---
# if __name__ == "__main__":
#     # Path to your test image and models
#     test_image_path = 'C:\\Users\\bss10\\Desktop\\drishti\\drishti_kavach\\number_plate_pic\\1.jpeg' 
    
#     # NOTE: You will need a YOLOv8 model trained on license plates. 
#     # For testing, replace with the path to your specific .pt file.
#     detector, reader = initialize_anpr_models('license_plate_detector.pt')
    
#     # Load the test image
#     image = cv2.imread(test_image_path)
    
#     if image is not None:
#         print("Processing image...")
#         processed_image, plate_data = read_license_plate(image, detector, reader)
        
#         # Print the extracted data
#         for i, data in enumerate(plate_data):
#             print(f"Plate {i+1}: {data['plate_text']} | OCR Conf: {data['ocr_confidence']:.2f} | YOLO Conf: {data['detection_confidence']:.2f}")
        
#         # Display the output
#         cv2.imshow('ANPR Output', processed_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("Error: Could not load the image.")


















# edit 2:
















# import cv2
# import easyocr
# import os
# from ultralytics import YOLO

# def initialize_anpr_models(yolo_model_path='yolov8n_plates.pt'):
#     """
#     Initializes both the YOLOv8 detector and the EasyOCR reader.
#     """
#     print("Loading YOLOv8 License Plate Model...")
#     # Load your custom fine-tuned YOLOv8 model for plates
#     plate_detector = YOLO(yolo_model_path)
    
#     print("Loading EasyOCR Reader...")
#     # Initialize EasyOCR for English text
#     # Setting gpu=False ensures it runs on CPU, ideal for a Raspberry Pi setup
#     ocr_reader = easyocr.Reader(['en'], gpu=False) 
    
#     return plate_detector, ocr_reader

# def read_license_plate(frame, plate_detector, ocr_reader):
#     """
#     Detects a license plate in a frame and extracts the text.
#     """
#     # 1. Run YOLOv8 inference to find the license plate
#     results = plate_detector(frame, conf=0.5)[0]
    
#     detected_plates = []

#     # 2. Iterate over the detections
#     for box in results.boxes:
#         # Extract the bounding box coordinates
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         confidence = float(box.conf[0])
        
#         # 3. Crop the detected license plate from the original frame
#         plate_crop = frame[y1:y2, x1:x2]
        
#         # Optional: Convert the crop to grayscale to improve OCR accuracy
#         gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
#         # 4. Pass the cropped image to EasyOCR to read the text
#         ocr_results = ocr_reader.readtext(gray_plate)
        
#         plate_text = ""
#         text_confidence = 0.0
        
#         # 5. Parse the OCR results (EasyOCR returns a list of tuples: [bbox, text, confidence])
#         # 5. Parse the OCR results
#         # 5. Parse the OCR results
#         if ocr_results:
#             # Group text blocks by line (rounding the Y-coordinate to nearest 15 pixels)
#             # Then sort left-to-right (by the X-coordinate) within that line
#             ocr_results.sort(key=lambda x: (round(x[0][0][1] / 15) * 15, x[0][0][0]))

#             # Extract the text from all blocks, remove spaces, and convert to uppercase
#             plate_text = "".join([result[1].replace(" ", "").upper() for result in ocr_results])
            
#             # Calculate the average confidence score across all blocks
#             total_confidence = sum([result[2] for result in ocr_results])
#             text_confidence = total_confidence / len(ocr_results)
            
#         # Store the structured data
#         detected_plates.append({
#             "bbox": (x1, y1, x2, y2),
#             "detection_confidence": confidence,
#             "plate_text": plate_text,
#             "ocr_confidence": text_confidence
#         })
        
#         # VISUALIZATION (Optional) 
#         # Draw the bounding box and text on the frame for visual confirmation
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{plate_text} ({text_confidence:.2f})", 
#                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.9, (0, 255, 0), 2)

#     return frame, detected_plates

# # EXECUTION 
# if __name__ == "__main__":
#     # Define the target folder path
#     folder_path = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\number_plate_pic\car images\train\images"
    
#     # Initialize models
#     detector, reader = initialize_anpr_models('best.pt')
    
#     # Create a resizable window and set a default starting size (Width, Height)
#     cv2.namedWindow('ANPR Output', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('ANPR Output', 800, 600) 
    
#     # Iterate through every file in the folder
#     for filename in os.listdir(folder_path):
        
#         # Verify the file is an image
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(folder_path, filename)
#             image = cv2.imread(image_path)
            
#             if image is not None:
#                 print(f"\nProcessing file: {filename}")
#                 processed_image, plate_data = read_license_plate(image, detector, reader)
                
#                 # Print the extracted data
#                 for i, data in enumerate(plate_data):
#                     print(f"Plate {i+1}: {data['plate_text']} | OCR Conf: {data['ocr_confidence']:.2f} | YOLO Conf: {data['detection_confidence']:.2f}")
                
#                 # Display the output in the resizable window
#                 cv2.imshow('ANPR Output', processed_image)
                
#                 # Wait for user input to proceed or quit
#                 key = cv2.waitKey(0) & 0xFF
#                 if key == 27 or key == ord('q'):
#                     print("Quitting process...")
#                     break
#             else:
#                 print(f"Error: Could not load the image {filename}.")
                
#     # Clean up windows when the script finishes or quits
#     cv2.destroyAllWindows()
















# edit 3:


















import cv2
import easyocr
import os
from ultralytics import YOLO

def initialize_anpr_models(yolo_model_path=r'C:\Users\bss10\Desktop\drishti\drishti_kavach\all_models\best (13).pt'):
    """
    Initializes both the YOLOv8 detector and the EasyOCR reader.
    """
    print("Loading YOLOv8 License Plate Model...")
    plate_detector = YOLO(yolo_model_path)
    
    print("Loading EasyOCR Reader...")
    ocr_reader = easyocr.Reader(['en'], gpu=False) 
    
    return plate_detector, ocr_reader

def read_license_plate(frame, plate_detector, ocr_reader):
    """
    Detects a license plate in a frame and extracts the text.
    """
    results = plate_detector(frame, conf=0.01)[0]
    
    detected_plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        
        plate_crop = frame[y1:y2, x1:x2]
        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        ocr_results = ocr_reader.readtext(gray_plate)
        
        plate_text = ""
        text_confidence = 0.0
        
        if ocr_results:
            ocr_results.sort(key=lambda x: (round(x[0][0][1] / 15) * 15, x[0][0][0]))
            plate_text = "".join([result[1].replace(" ", "").upper() for result in ocr_results])
            
            total_confidence = sum([result[2] for result in ocr_results])
            text_confidence = total_confidence / len(ocr_results)
            
        # EXCLUSION RULE: Skip if OCR confidence is exactly 0.00
        if text_confidence == 0.0:
            continue
            
        detected_plates.append({
            "bbox": (x1, y1, x2, y2),
            "detection_confidence": confidence,
            "plate_text": plate_text,
            "ocr_confidence": text_confidence
        })
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{plate_text} ({text_confidence:.2f})", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)

    return frame, detected_plates

if __name__ == "__main__":
    folder_path = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\datasets\Indian License Plate Detection.yolov8\test\images"
    
    detector, reader = initialize_anpr_models('best.pt')
    
    cv2.namedWindow('ANPR Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ANPR Output', 800, 600) 
    
    for filename in os.listdir(folder_path):
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                print(f"\nProcessing file: {filename}")
                processed_image, plate_data = read_license_plate(image, detector, reader)
                
                for i, data in enumerate(plate_data):
                    print(f"Plate {i+1}: {data['plate_text']} | OCR Conf: {data['ocr_confidence']:.2f} | YOLO Conf: {data['detection_confidence']:.2f}")
                
                cv2.imshow('ANPR Output', processed_image)
                
                key = cv2.waitKey(0) & 0xFF
                if key == 27 or key == ord('q'):
                    print("Quitting process...")
                    break
            else:
                print(f"Error: Could not load the image {filename}.")
                
    cv2.destroyAllWindows()




















# edit 4:




















# import cv2
# import easyocr
# import os
# import re
# from ultralytics import YOLO

# def initialize_anpr_models(yolo_model_path=r'C:\Users\bss10\Desktop\drishti\drishti_kavach\best (1).pt'):
#     """
#     Initializes both the YOLOv8 detector and the EasyOCR reader.
#     """
#     print("Loading YOLOv8 License Plate Model...")
#     plate_detector = YOLO(yolo_model_path)
    
#     print("Loading EasyOCR Reader...")
#     ocr_reader = easyocr.Reader(['en'], gpu=False) 
    
#     return plate_detector, ocr_reader

# def read_license_plate(frame, plate_detector, ocr_reader):
#     """
#     Diagnostic mode: Detects everything the model thinks might be a plate.
#     """
#     # Replace the old conf=0.5 line with this one:
#     results = plate_detector(frame, conf=0.1)[0]
    
#     detected_plates = []
    
#     print(f"YOLO found {len(results.boxes)} raw bounding boxes.")

#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         confidence = float(box.conf[0])
        
#         # Draw every single box YOLO finds in red, regardless of quality
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     return frame, detected_plates

# if __name__ == "__main__":
#     folder_path = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\number_plate_pic\car images\valid\images"
    
#     detector, reader = initialize_anpr_models('best.pt')
    
#     cv2.namedWindow('ANPR Output', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('ANPR Output', 800, 600) 
    
#     for filename in os.listdir(folder_path):
        
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(folder_path, filename)
#             image = cv2.imread(image_path)
            
#             if image is not None:
#                 print(f"\nProcessing file: {filename}")
#                 processed_image, plate_data = read_license_plate(image, detector, reader)
                
#                 if not plate_data:
#                     print("No valid plates detected matching the criteria.")
#                 else:
#                     for i, data in enumerate(plate_data):
#                         print(f"Valid Plate {i+1}: {data['plate_text']} | OCR Conf: {data['ocr_confidence']:.2f} | YOLO Conf: {data['detection_confidence']:.2f}")
                
#                 cv2.imshow('ANPR Output', processed_image)
                
#                 key = cv2.waitKey(0) & 0xFF
#                 if key == 27 or key == ord('q'):
#                     print("Quitting process...")
#                     break
#             else:
#                 print(f"Error: Could not load the image {filename}.")
                
#     cv2.destroyAllWindows()














# edit 5:
# with roboflow api












# import time
# import cv2
# import easyocr
# import os
# from inference_sdk import InferenceHTTPClient

# def initialize_anpr_models():
#     """
#     Initializes the Roboflow client and the EasyOCR reader.
#     """
#     print("Loading Roboflow Inference Client...")
#     # Connects to your local inference server
#     plate_detector = InferenceHTTPClient(
#         api_url="https://detect.roboflow.com", 
#         api_key="Fi7vB4C5wgI6q5s9FSVT"
#     )
    
#     print("Loading EasyOCR Reader...")
#     ocr_reader = easyocr.Reader(['en'], gpu=False) 
    
#     return plate_detector, ocr_reader

# def read_license_plate(frame, plate_detector, ocr_reader):
#     """
#     Detects a license plate in a frame and extracts the text.
#     """
#     # Run inference using your specified project and version
#     results = plate_detector.infer(frame, model_id="license-plate-recognition-rxg4e/11")
    
#     detected_plates = []
    
#     # Roboflow stores the bounding boxes in the 'predictions' list
#     predictions = results.get("predictions", [])

#     for pred in predictions:
#         confidence = float(pred["confidence"])
        
#         # Roboflow returns center points, width, and height. 
#         # Convert these to x1, y1, x2, y2 coordinates.
#         x_center = pred["x"]
#         y_center = pred["y"]
#         width = pred["width"]
#         height = pred["height"]
        
#         x1 = int(x_center - (width / 2))
#         y1 = int(y_center - (height / 2))
#         x2 = int(x_center + (width / 2))
#         y2 = int(y_center + (height / 2))
        
#         # Prevent negative coordinates if the box hits the image edge
#         x1, y1 = max(0, x1), max(0, y1)
        
#         plate_crop = frame[y1:y2, x1:x2]
        
#         # Skip if the crop is empty
#         if plate_crop.size == 0:
#             continue
            
#         gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
#         ocr_results = ocr_reader.readtext(gray_plate)
        
#         plate_text = ""
#         text_confidence = 0.0
        
#         if ocr_results:
#             ocr_results.sort(key=lambda x: (round(x[0][0][1] / 15) * 15, x[0][0][0]))
#             plate_text = "".join([result[1].replace(" ", "").upper() for result in ocr_results])
            
#             total_confidence = sum([result[2] for result in ocr_results])
#             text_confidence = total_confidence / len(ocr_results)
            
#         # Skip if OCR confidence is exactly 0.00
#         if text_confidence == 0.0:
#             continue
            
#         detected_plates.append({
#             "bbox": (x1, y1, x2, y2),
#             "detection_confidence": confidence,
#             "plate_text": plate_text,
#             "ocr_confidence": text_confidence
#         })
        
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{plate_text} ({text_confidence:.2f})", 
#                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.9, (0, 255, 0), 2)

#     return frame, detected_plates

# if __name__ == "__main__":
#     folder_path = r"C:\Users\bss10\Desktop\drishti\drishti_kavach\number_plate_pic\car images\valid\images"
    
#     detector, reader = initialize_anpr_models()
    
#     cv2.namedWindow('ANPR Output', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('ANPR Output', 800, 600) 
    
#     for filename in os.listdir(folder_path):
        
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(folder_path, filename)
#             image = cv2.imread(image_path)
            
#             if image is not None:
#                 print(f"\nProcessing file: {filename}")
#                 processed_image, plate_data = read_license_plate(image, detector, reader)
                
#                 for i, data in enumerate(plate_data):
#                     print(f"Plate {i+1}: {data['plate_text']} | OCR Conf: {data['ocr_confidence']:.2f} | Roboflow Conf: {data['detection_confidence']:.2f}")
                
#                 cv2.imshow('ANPR Output', processed_image)
                
#                 key = cv2.waitKey(0) & 0xFF
#                 if key == 27 or key == ord('q'):
#                     print("Quitting process...")
#                     break
#                 time.sleep(1)
#             else:
#                 print(f"Error: Could not load the image {filename}.")
                
#     cv2.destroyAllWindows()