import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class PizzaSegmentationService:
    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')
        self.pizza_class_id = 53
    
    def segment_pizza(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        results = self.model(image, verbose=False)
        
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for result in results:
            if result.masks is not None and result.boxes is not None:
                for i, (box, mask_data) in enumerate(zip(result.boxes, result.masks)):
                    class_id = int(box.cls)
                    
                    if class_id == self.pizza_class_id:
                        pizza_mask = mask_data.data[0].cpu().numpy()
                        pizza_mask_resized = cv2.resize(
                            pizza_mask, 
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                        mask[pizza_mask_resized > 0.5] = 255
        
        return mask
    
    def save_mask(self, mask: np.ndarray, output_path: str):
        cv2.imwrite(output_path, mask)


if __name__ == "__main__":
    service = PizzaSegmentationService()
    
    input_image = "resource/pizza.jpg"
    output_mask = "result/pizza_mask.png"
    
    try:
        mask = service.segment_pizza(input_image)
        
        print(f"Mask shape: {mask.shape}")
        print(f"Unique values in mask: {np.unique(mask)}")
        print(f"Number of white pixels: {np.sum(mask == 255)}")
        
        service.save_mask(mask, output_mask)
        print(f"Successfully saved segmentation mask to {output_mask}")
    except Exception as e:
        print(f"Error: {e}")