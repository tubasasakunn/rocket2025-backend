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
            
            # In cv2.fitEllipse: width corresponds to the first axis, height to the second axis
            # The angle represents the rotation of the first axis from horizontal
            # We need to ensure major_axis is always the longer one and adjust angle accordingly
            if width >= height:
                # width is major axis, height is minor axis
                major_axis = width / 2
                minor_axis = height / 2
                # angle already represents the major axis angle
                major_axis_angle = angle
            else:
                # height is major axis, width is minor axis
                major_axis = height / 2
                minor_axis = width / 2
                # major axis is perpendicular to the angle given
                major_axis_angle = angle + 90
                if major_axis_angle >= 180:
                    major_axis_angle -= 180
            
            return ((int(center_x), int(center_y)), (int(major_axis), int(minor_axis)), major_axis_angle)
        
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
        Only scale the x-axis (major axis) to match the minor axis
        
        Args:
            major_axis: Length of major axis (will be horizontal after rotation)
            minor_axis: Length of minor axis (will be vertical after rotation)
            
        Returns:
            (scale_x, scale_y) factors
        """
        # Scale only x-axis (major axis) to match y-axis (minor axis)
        # This makes the ellipse circular by shrinking the longer dimension
        scale_x = minor_axis / major_axis  # Shrink x-axis to match y-axis
        scale_y = 1.0  # Keep y-axis unchanged
        
        return (scale_x, scale_y)
    
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
        scale_x, scale_y = self.calculate_transform_scale(major_axis, minor_axis)
        
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
        # Only scale x-axis (shrink major axis to match minor axis)
        M3 = np.array([[scale_x, 0, 0],
                       [0, scale_y, 0],
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
    
    def transform_image_step_by_step(self, image: np.ndarray, ellipse_params: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform image step by step for debugging
        
        Args:
            image: Input image
            ellipse_params: ((center_x, center_y), (major_axis, minor_axis), angle)
            
        Returns:
            (rotated_image, scaled_image, final_image)
        """
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
        
        # Calculate scale factors
        scale_x, scale_y = self.calculate_transform_scale(major_axis, minor_axis)
        
        # Step 1: Only rotation (align ellipse with axes)
        M1 = np.array([[1, 0, -center_x],
                       [0, 1, -center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # Rotate to align major axis with x-axis
        angle_rad = np.radians(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        M2 = np.array([[cos_a, -sin_a, 0],
                       [sin_a, cos_a, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        M5 = np.array([[1, 0, center_x],
                       [0, 1, center_y],
                       [0, 0, 1]], dtype=np.float32)
        
        # Only rotation transformation (keep ellipse aligned with axes)
        M_rotation = M5 @ M2 @ M1
        M_rotation = M_rotation[:2, :]
        rotated = cv2.warpAffine(image, M_rotation, (image.shape[1], image.shape[0]))
        
        # Step 2: Add scaling to the rotated image
        # Only scale x-axis (shrink major axis to match minor axis)
        M3 = np.array([[scale_x, 0, 0],
                       [0, scale_y, 0],
                       [0, 0, 1]], dtype=np.float32)
        
        # Full transformation: rotate then scale (no rotation back)
        M_full = M5 @ M3 @ M2 @ M1
        M_full = M_full[:2, :]
        final = cv2.warpAffine(image, M_full, (image.shape[1], image.shape[0]))
        
        return rotated, final, final
    
    def normalize_to_512x512(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        """
        Crop pizza by radius and resize to 512x512
        
        Args:
            image: Input image (should be transformed to circular)
            center: Center coordinates (x, y)
            radius: Pizza radius for cropping
            
        Returns:
            512x512 normalized image
        """
        center_x, center_y = center
        
        # Calculate crop bounds
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(image.shape[1], center_x + radius)
        y2 = min(image.shape[0], center_y + radius)
        
        # Crop the image
        cropped = image[y1:y2, x1:x2]
        
        # Resize to 512x512
        normalized = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        return normalized
    
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
        info['transformation_applied'] = {
            'scale_x': minor_axis / major_axis,  # Shrink x-axis (major axis)
            'scale_y': 1.0                       # Keep y-axis unchanged
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
        try:
            print(f"\nProcessing {input_path}...")
            
            # Read image
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"Could not read image from {input_path}")
                continue
            
            # Get segmentation mask and ellipse parameters
            mask = service.segmentation_service.segment_pizza(str(input_path))
            ellipse_params = service.detect_ellipse_from_mask(mask)
            
            if ellipse_params is None:
                print("No ellipse detected in the image")
                continue
            
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse_params
            print(f"Detected ellipse: Center=({center_x}, {center_y}), Major={major_axis}, Minor={minor_axis}, Angle={angle:.2f}°")
            
            # Step-by-step transformation
            rotated, final, _ = service.transform_image_step_by_step(image, ellipse_params)
            
            # 3. Normalize to 512x512 (crop by pizza radius and resize)
            # Use minor_axis as radius since pizza is now circular
            normalized = service.normalize_to_512x512(final, (center_x, center_y), minor_axis)
            
            # Save step-by-step results
            base_name = input_path.stem
            
            # 0. Save original
            original_path = f"debug/preprocess/{base_name}_original.jpg"
            Path(original_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(original_path, image)
            print(f"  ✓ Saved original image to {original_path}")
            
            # 1. Save rotated only
            rotated_path = f"debug/preprocess/{base_name}_rotated.jpg"
            cv2.imwrite(rotated_path, rotated)
            print(f"  ✓ Saved rotated image to {rotated_path}")
            
            # 2. Save final (rotated + scaled)
            final_path = f"debug/preprocess/{base_name}_final.jpg"
            cv2.imwrite(final_path, final)
            print(f"  ✓ Saved final image to {final_path}")
            
            # 3. Save normalized 512x512
            normalized_path = f"debug/preprocess/{base_name}_normalized.jpg"
            cv2.imwrite(normalized_path, normalized)
            print(f"  ✓ Saved normalized 512x512 image to {normalized_path}")
            
            # Calculate and print scale factors
            scale_x = minor_axis / major_axis  # Shrink x-axis only
            scale_y = 1.0  # Keep y-axis unchanged
            print(f"  Scale factors: x={scale_x:.4f} (shrink major axis), y={scale_y:.4f} (unchanged)")
            print(f"  Normalized: cropped by radius {minor_axis}px and resized to 512x512")
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")