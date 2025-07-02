import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pizza_segmentation_service import PizzaSegmentationService
from salami_segmentation_service import SalamiSegmentationService


class SalamiCircleDetectionService:
    def __init__(self):
        self.pizza_service = PizzaSegmentationService()
        self.salami_service = SalamiSegmentationService()
    
    def detect_circles_from_mask(self, mask: np.ndarray) -> List[Tuple[Tuple[int, int], int]]:
        """
        Detect circles from a binary mask using contour analysis and minimum enclosing circles
        
        Args:
            mask: Binary mask (0 or 255 values)
            
        Returns:
            List of tuples, each containing ((center_x, center_y), radius)
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for contour in contours:
            # Calculate area to filter out small noise
            area = cv2.contourArea(contour)
            if area < 200:  # Lower threshold to catch smaller salami
                continue
            
            # Find minimum enclosing circle for each contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Calculate circularity to filter out non-circular shapes
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Balanced thresholds to detect individual salami
                if circularity > 0.3 and radius > 12:  # More lenient for better detection
                    circles.append(((int(x), int(y)), int(radius)))
        
        return circles
    
    def detect_salami_circles(self, image_path: str) -> List[Tuple[Tuple[int, int], int]]:
        """
        Detect salami circles from an image by combining pizza and salami segmentation
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of tuples, each containing ((center_x, center_y), radius)
        """
        # Get salami mask
        salami_mask = self.salami_service.segment_salami(image_path)
        
        # Get pizza mask
        pizza_mask = self.pizza_service.segment_pizza(image_path)
        
        # Multiply masks (bitwise AND operation) to get salami only within pizza
        final_mask = cv2.bitwise_and(salami_mask, pizza_mask)
        
        # First pass: moderate erosion
        separation_kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        final_mask = cv2.erode(final_mask, separation_kernel1, iterations=1)
        
        # Second pass: targeted erosion for stubborn connections
        separation_kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        final_mask = cv2.erode(final_mask, separation_kernel2, iterations=1)
        
        # Dilate back carefully
        recovery_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        final_mask = cv2.dilate(final_mask, recovery_kernel, iterations=1)
        
        # Clean up small noise
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, small_kernel)
        
        # Detect circles from the cleaned mask
        circles = self.detect_circles_from_mask(final_mask)
        
        return circles
    
    def draw_circles_on_image(self, image: np.ndarray, circles: List[Tuple[Tuple[int, int], int]]) -> np.ndarray:
        """
        Draw detected circles on the image
        
        Args:
            image: Input image
            circles: List of circles to draw
            
        Returns:
            Image with circles drawn
        """
        result = image.copy()
        
        for (center_x, center_y), radius in circles:
            # Draw circle outline
            cv2.circle(result, (center_x, center_y), radius, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)
            # Add text with circle info
            text = f"({center_x}, {center_y}) r={radius}"
            cv2.putText(result, text, (center_x - 50, center_y - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return result


if __name__ == "__main__":
    # Create service instance
    service = SalamiCircleDetectionService()
    
    # Input and output paths
    input_image = "resource/pizza.jpg"
    output_image = "result/salami_circles_overlay.jpg"
    
    try:
        # Read original image
        image = cv2.imread(input_image)
        if image is None:
            raise ValueError(f"Could not read image from {input_image}")
        
        # Detect salami circles
        print("Detecting salami circles...")
        circles = service.detect_salami_circles(input_image)
        
        print(f"Detected {len(circles)} salami circles:")
        for i, ((cx, cy), r) in enumerate(circles):
            print(f"  Circle {i+1}: Center=({cx}, {cy}), Radius={r}")
        
        # Draw circles on the original image
        result_image = service.draw_circles_on_image(image, circles)
        
        # Save the result
        Path(output_image).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_image, result_image)
        print(f"Successfully saved overlay image to {output_image}")
        
    except Exception as e:
        print(f"Error: {e}")