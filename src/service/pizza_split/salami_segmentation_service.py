import cv2
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pizza_segmentation_service import PizzaSegmentationService


class SalamiSegmentationService:
    # Magic numbers for salami detection
    COLOR_QUANTIZATION_K = 6  # Number of colors for K-means clustering
    HUE_LOWER_BOUND = 0  # Lower bound for red hue (0-10 or 170-180)
    HUE_UPPER_BOUND_1 = 8  # Upper bound for first red range
    HUE_LOWER_BOUND_2 = 172  # Lower bound for second red range
    HUE_UPPER_BOUND_2 = 180  # Upper bound for second red range
    SATURATION_THRESHOLD = 120  # Minimum saturation for salami
    VALUE_THRESHOLD = 90  # Minimum value (brightness) for salami
    MORPHOLOGY_KERNEL_SIZE = 5  # Kernel size for morphological operations
    BILATERAL_D = 15  # Diameter of bilateral filter
    BILATERAL_SIGMA_COLOR = 80  # Color sigma for bilateral filter
    BILATERAL_SIGMA_SPACE = 80  # Space sigma for bilateral filter
    MIN_CONTOUR_AREA = 1000  # Minimum area for salami pieces
    THIN_LINE_KERNEL_SIZE = 7  # Kernel size for removing thin lines
    
    def __init__(self):
        pass
    
    def quantize_colors(self, image: np.ndarray, k: int) -> np.ndarray:
        """Apply K-means clustering to reduce colors in the image"""
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized_image = quantized_data.reshape(image.shape)
        
        return quantized_image
    
    def detect_salami_by_color(self, image: np.ndarray) -> np.ndarray:
        """Detect salami regions based on color (red hue)"""
        # Apply bilateral filter for edge-preserving smoothing
        smoothed = cv2.bilateralFilter(image, self.BILATERAL_D, self.BILATERAL_SIGMA_COLOR, self.BILATERAL_SIGMA_SPACE)
        
        # Apply color quantization
        quantized = self.quantize_colors(smoothed, self.COLOR_QUANTIZATION_K)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color ranges
        # Red wraps around in HSV, so we need two ranges
        lower_red1 = np.array([self.HUE_LOWER_BOUND, self.SATURATION_THRESHOLD, self.VALUE_THRESHOLD])
        upper_red1 = np.array([self.HUE_UPPER_BOUND_1, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([self.HUE_LOWER_BOUND_2, self.SATURATION_THRESHOLD, self.VALUE_THRESHOLD])
        upper_red2 = np.array([self.HUE_UPPER_BOUND_2, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks
        salami_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((self.MORPHOLOGY_KERNEL_SIZE, self.MORPHOLOGY_KERNEL_SIZE), np.uint8)
        salami_mask = cv2.morphologyEx(salami_mask, cv2.MORPH_CLOSE, kernel)
        salami_mask = cv2.morphologyEx(salami_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small contours
        contours, _ = cv2.findContours(salami_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(salami_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.MIN_CONTOUR_AREA:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
    def segment_salami(self, image_path: str) -> np.ndarray:
        """Main method to segment salami from pizza image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        return self.detect_salami_by_color(image)
    
    def save_mask(self, mask: np.ndarray, output_path: str):
        """Save the segmentation mask"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, mask)


if __name__ == "__main__":
    salami_service = SalamiSegmentationService()
    pizza_service = PizzaSegmentationService()
    
    input_image = "resource/pizza.jpg"
    output_mask = "result/salami_mask.png"
    
    try:
        # Get salami mask
        salami_mask = salami_service.segment_salami(input_image)
        
        # Get pizza mask
        pizza_mask = pizza_service.segment_pizza(input_image)
        
        # Multiply masks (bitwise AND operation)
        final_mask = cv2.bitwise_and(salami_mask, pizza_mask)
        
        # Remove thin lines using morphological opening with larger kernel
        thin_line_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (salami_service.THIN_LINE_KERNEL_SIZE, 
                                                      salami_service.THIN_LINE_KERNEL_SIZE))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, thin_line_kernel)
        
        # Additional erosion and dilation to clean up
        final_mask = cv2.erode(final_mask, thin_line_kernel, iterations=1)
        final_mask = cv2.dilate(final_mask, thin_line_kernel, iterations=1)
        
        print(f"Salami mask shape: {salami_mask.shape}")
        print(f"Pizza mask shape: {pizza_mask.shape}")
        print(f"Final mask shape: {final_mask.shape}")
        print(f"Unique values in final mask: {np.unique(final_mask)}")
        print(f"Number of white pixels in salami mask: {np.sum(salami_mask == 255)}")
        print(f"Number of white pixels in pizza mask: {np.sum(pizza_mask == 255)}")
        print(f"Number of white pixels in final mask: {np.sum(final_mask == 255)}")
        
        salami_service.save_mask(final_mask, output_mask)
        print(f"Successfully saved salami segmentation mask (within pizza bounds) to {output_mask}")
    except Exception as e:
        print(f"Error: {e}")