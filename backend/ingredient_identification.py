import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from typing import List, Tuple, Union
import numpy as np

MODEL_PATH = os.path.join("models", "model1.pt") 

# Loading the fine-tuned yolov8m model 
model = None
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLOv8 model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")

def identify_ingredients(image_path: str) -> Tuple[List[str], Union[np.ndarray, None]]:
    if model is None:
        print("YOLOv8 model not loaded.")
        return [], None
    
    try:
        # read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return [], None
        
        results = model(image, conf=0.25)
        
       
        detected_ingredients = list(set(results[0].names[int(i)] for i in results[0].boxes.cls))
        print(f"Detected ingredients: {detected_ingredients}")
        
       
        annotated_image = results[0].plot()  # plot detections on the image
        
        # save the annotated image
        output_dir = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        file_name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{file_name}_annotated{ext}")
        cv2.imwrite(output_path, annotated_image)
        
        return detected_ingredients, annotated_image
        
    except Exception as e:
        print(f"Error during YOLOv8 inference: {e}")
        return [], None

if __name__ == '__main__':
    test_image_path = "assets//demo_non_veg2.png"  
    if os.path.exists(test_image_path):
        ingredients, annotated_img = identify_ingredients(test_image_path)
        print(f"Identified ingredients in {test_image_path}: {ingredients}")
        
        if annotated_img is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Detected Ingredients")
            plt.show()
    else:
        print(f"Test image not found at {test_image_path}")