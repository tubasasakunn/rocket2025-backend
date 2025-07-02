import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pizza_segmentation_service import PizzaSegmentationService


class PreprocessService:
    def __init__(self):
        self.segmentation_service = PizzaSegmentationService()
        self.ellipse_threshold = 0.1  # 10% difference between major and minor axes
    
    def detect_ellipse_from_mask(self, mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """
        Detect ellipse from segmentation mask
        
        Args:
            mask: Binary mask (255 = pizza, 0 = background)
            
        Returns:
            ((center_x, center_y), (major_axis, minor_axis), angle) or None
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit ellipse if contour has enough points
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (center_x, center_y), (width, height), angle = ellipse
            
            # Convert to major and minor axes
            major_axis = max(width, height) / 2
            minor_axis = min(width, height) / 2
            
            return ((int(center_x), int(center_y)), (int(major_axis), int(minor_axis)), angle)
        
        return None
    
    def is_elliptical(self, major_axis: int, minor_axis: int) -> bool:
        """
        Check if the shape is elliptical based on axis ratio
        
        Args:
            major_axis: Length of major axis
            minor_axis: Length of minor axis
            
        Returns:
            True if elliptical, False if circular
        """
        if minor_axis == 0:
            return True
        
        ratio = (major_axis - minor_axis) / major_axis
        return ratio > self.ellipse_threshold
    
    def calculate_transform_scale(self, major_axis: int, minor_axis: int) -> Tuple[float, float]:
        """
        Calculate scale factors to transform ellipse to circle
        
        Args:
            major_axis: Length of major axis
            minor_axis: Length of minor axis
            
        Returns:
            (scale_major, scale_minor) factors
        """
        # Target radius is the average of major and minor axes
        target_radius = (major_axis + minor_axis) / 2
        
        # Calculate scale factors
        # scale_major < 1.0 (shrink major axis)
        # scale_minor > 1.0 (expand minor axis)
        scale_major = target_radius / major_axis  # Shrink long axis
        scale_minor = target_radius / minor_axis  # Expand short axis
        
        return (scale_major, scale_minor)
    
    def transform_image_to_circular(self, image: np.ndarray, ellipse_params: Tuple) -> np.ndarray:
        """
        Transform image to make elliptical pizza circular
        
        Args:
            image: Input image
            ellipse_params: ((center_x, center_y), (major_axis, minor_axis), angle)
            
        Returns:
            Transformed image
        """
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
        
        # Calculate scale factors
        scale_major, scale_minor = self.calculate_transform_scale(major_axis, minor_axis)
        
        # Create transformation matrix
        # First, translate to origin
        M1 = np.array([[1, 0, -center_x],
                       [0, 1, -center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # Rotate to align with axes (inverse rotation)
        angle_rad = np.radians(-angle)  # Negate angle for inverse rotation
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        M2 = np.array([[cos_a, -sin_a, 0],
                       [sin_a, cos_a, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # Scale to make circular
        # After rotation, major axis is aligned with x-axis, minor axis with y-axis
        M3 = np.array([[scale_major, 0, 0],
                       [0, scale_minor, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # Rotate back (forward rotation)
        angle_rad_back = np.radians(angle)  # Original angle for rotating back
        cos_b = np.cos(angle_rad_back)
        sin_b = np.sin(angle_rad_back)
        M4 = np.array([[cos_b, -sin_b, 0],
                       [sin_b, cos_b, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # Translate back
        M5 = np.array([[1, 0, center_x],
                       [0, 1, center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # Combine transformations
        M = M5 @ M4 @ M3 @ M2 @ M1
        M = M[:2, :]  # Remove homogeneous coordinate
        
        # Apply transformation
        transformed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        return transformed
    
    def preprocess_pizza_image(self, image_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """
        Main preprocessing function
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save preprocessed image
            
        Returns:
            (preprocessed_image, info_dict)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Get segmentation mask
        mask = self.segmentation_service.segment_pizza(image_path)
        
        # Detect ellipse
        ellipse_params = self.detect_ellipse_from_mask(mask)
        
        info = {
            'original_shape': image.shape,
            'is_transformed': False,
            'ellipse_params': None,
            'transformation_applied': None
        }
        
        if ellipse_params is None:
            print("No ellipse detected in the image")
            return image, info
        
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
        info['ellipse_params'] = {
            'center': (center_x, center_y),
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle
        }
        
        print(f"Detected ellipse: Center=({center_x}, {center_y}), Major={major_axis}, Minor={minor_axis}, Angle={angle:.2f}°")
        
        # Always apply transformation (treat all pizzas as elliptical)
        print(f"Applying transformation to pizza: Center=({center_x}, {center_y}), Major={major_axis}, Minor={minor_axis}, Angle={angle:.2f}°")
        
        # Transform image
        transformed = self.transform_image_to_circular(image, ellipse_params)
        
        info['is_transformed'] = True
        target_radius = (major_axis + minor_axis) / 2
        info['transformation_applied'] = {
            'scale_major': target_radius / major_axis,  # Should be < 1.0 (shrink)
            'scale_minor': target_radius / minor_axis   # Should be > 1.0 (expand)
        }
        
        # Save if output path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, transformed)
            print(f"Saved preprocessed image to {output_path}")
        
        return transformed, info


if __name__ == "__main__":
    # Create service instance
    service = PreprocessService()
    
    # Find all images in resource directory
    resource_dir = Path("resource")
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    input_images = []
    
    for ext in supported_extensions:
        input_images.extend(resource_dir.glob(f"*{ext}"))
        input_images.extend(resource_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(input_images)} images in resource directory")
    
    for input_path in input_images:
        output_path = f"debug/preprocess/{input_path.name}"
        
        try:
            print(f"\nProcessing {input_path}...")
            preprocessed_image, info = service.preprocess_pizza_image(str(input_path), output_path)
            
            # Print transformation info
            if info['is_transformed']:
                print("Transformation applied successfully")
                print(f"  Original axes: {info['ellipse_params']['major_axis']} x {info['ellipse_params']['minor_axis']}")
                print(f"  Scale factors: {info['transformation_applied']}")
            else:
                print("No transformation was necessary")
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")