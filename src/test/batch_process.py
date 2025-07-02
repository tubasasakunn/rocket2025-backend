import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import services
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.pizza_circle_detection_service import PizzaCircleDetectionService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.salami_circle_detection_service import SalamiCircleDetectionService


def process_all_images():
    """Process all images in resource/ directory"""
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    sample_dir = base_dir / "resource"
    result_dir = base_dir / "result"
    
    # Initialize services
    pizza_seg_service = PizzaSegmentationService()
    pizza_circle_service = PizzaCircleDetectionService()
    salami_seg_service = SalamiSegmentationService()
    salami_circle_service = SalamiCircleDetectionService()
    
    # Process names and corresponding services
    processes = {
        "pizza_segmentation": {
            "service": pizza_seg_service,
            "process": lambda service, img_path: service.segment_pizza(img_path),
            "save": lambda mask, output_path: cv2.imwrite(output_path, mask)
        },
        "pizza_circle_detection": {
            "service": pizza_circle_service,
            "process": lambda service, img_path: process_pizza_circle(service, img_path),
            "save": lambda result, output_path: cv2.imwrite(output_path, result)
        },
        "salami_segmentation": {
            "service": salami_seg_service,
            "process": lambda service, img_path: process_salami_segmentation(service, img_path, pizza_seg_service),
            "save": lambda mask, output_path: cv2.imwrite(output_path, mask)
        },
        "salami_circle_detection": {
            "service": salami_circle_service,
            "process": lambda service, img_path: process_salami_circles(service, img_path),
            "save": lambda result, output_path: cv2.imwrite(output_path, result)
        }
    }
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(sample_dir.glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image with each service
    for img_path in sorted(image_files):
        print(f"\nProcessing: {img_path.name}")
        
        for process_name, process_info in processes.items():
            try:
                # Create output directory
                output_dir = result_dir / process_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process image
                result = process_info["process"](process_info["service"], str(img_path))
                
                # Save result
                output_path = output_dir / f"{img_path.stem}.png"
                process_info["save"](result, str(output_path))
                
                print(f"  ✓ {process_name} -> {output_path.relative_to(base_dir)}")
                
            except Exception as e:
                print(f"  ✗ {process_name} failed: {e}")


def process_pizza_circle(service, image_path):
    """Process pizza circle detection"""
    circle_info = service.detect_circle_from_image(image_path)
    
    if circle_info is None:
        # Return blank image if no circle detected
        img = cv2.imread(image_path)
        return img if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Read original image and draw circle
    original_image = cv2.imread(image_path)
    center, radius = circle_info
    result_image = service.draw_circle_on_image(original_image, center, radius)
    
    return result_image


def process_salami_segmentation(salami_service, image_path, pizza_service):
    """Process salami segmentation within pizza bounds"""
    # Get salami mask
    salami_mask = salami_service.segment_salami(image_path)
    
    # Get pizza mask
    pizza_mask = pizza_service.segment_pizza(image_path)
    
    # Multiply masks (bitwise AND operation)
    final_mask = cv2.bitwise_and(salami_mask, pizza_mask)
    
    # Clean up the mask
    thin_line_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (salami_service.THIN_LINE_KERNEL_SIZE, salami_service.THIN_LINE_KERNEL_SIZE)
    )
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, thin_line_kernel)
    final_mask = cv2.erode(final_mask, thin_line_kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, thin_line_kernel, iterations=1)
    
    return final_mask


def process_salami_circles(service, image_path):
    """Process salami circle detection"""
    # Read original image
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Detect salami circles
    circles = service.detect_salami_circles(image_path)
    
    # Draw circles on the original image
    result_image = service.draw_circles_on_image(image, circles)
    
    return result_image


if __name__ == "__main__":
    print("Starting batch processing of pizza images...")
    process_all_images()
    print("\nBatch processing completed!")