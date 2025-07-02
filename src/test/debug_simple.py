#!/usr/bin/env python3
"""
Simple debug script for testing the pizza processing pipeline.
This script uses the existing services from the pizza_split directory.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path to access service modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all services
from service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from service.pizza_split.pizza_circle_detection_service import PizzaCircleDetectionService
from service.pizza_split.salami_segmentation_service import SalamiSegmentationService
from service.pizza_split.salami_circle_detection_service import SalamiCircleDetectionService
from service.pizza_split.preprocess import PreprocessService


def main():
    print("=== Pizza Processing Pipeline Debug ===\n")
    
    # Initialize services
    pizza_seg_service = PizzaSegmentationService()
    pizza_circle_service = PizzaCircleDetectionService()
    salami_seg_service = SalamiSegmentationService()
    salami_circle_service = SalamiCircleDetectionService()
    preprocess_service = PreprocessService()
    
    # Set up directories
    project_root = Path(__file__).parent.parent.parent  # src/test -> src -> root
    resource_dir = project_root / "resource"
    debug_dir = project_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    image_files = list(resource_dir.glob("*.jpg")) + list(resource_dir.glob("*.jpeg"))
    
    if not image_files:
        print("No image files found in resource directory")
        return
    
    print(f"Found {len(image_files)} image files\n")
    
    # Process each image
    for image_path in image_files:
        print(f"Processing {image_path.name}...")
        
        try:
            # Step 1: Preprocess the image (convert elliptical pizzas to circular)
            preprocessed_path = debug_dir / f"preprocessed_{image_path.name}"
            preprocessed_image, info = preprocess_service.preprocess_pizza_image(
                str(image_path), str(preprocessed_path)
            )
            
            if info['is_transformed']:
                print(f"  ✓ Applied ellipse-to-circle transformation")
                working_image_path = str(preprocessed_path)
            else:
                print(f"  - No transformation needed (already circular)")
                working_image_path = str(image_path)
            
            # Step 2: Detect pizza circle
            print("  - Detecting pizza circle...")
            pizza_circle = pizza_circle_service.detect_circle_from_image(working_image_path)
            if pizza_circle:
                center, radius = pizza_circle
                print(f"  ✓ Pizza circle detected: center={center}, radius={radius}")
                
                # Draw pizza circle on image
                display_image = cv2.imread(working_image_path)
                pizza_circle_image = pizza_circle_service.draw_circle_on_image(
                    display_image.copy(), center, radius
                )
                cv2.imwrite(str(debug_dir / f"pizza_circle_{image_path.name}"), pizza_circle_image)
            else:
                print("  ✗ Failed to detect pizza circle")
                continue
            
            # Step 3: Detect salami circles
            print("  - Detecting salami circles...")
            salami_circles = salami_circle_service.detect_salami_circles(working_image_path)
            print(f"  ✓ Found {len(salami_circles)} salami circles")
            
            if salami_circles:
                # Draw salami circles on image
                salami_circle_image = salami_circle_service.draw_circles_on_image(
                    display_image.copy(), salami_circles
                )
                cv2.imwrite(str(debug_dir / f"salami_circles_{image_path.name}"), salami_circle_image)
                
                # Create combined visualization
                combined_image = display_image.copy()
                # Draw pizza circle in green
                cv2.circle(combined_image, (int(center[0]), int(center[1])), 
                          int(radius), (0, 255, 0), 2)
                # Draw salami circles in red
                for (center_s, radius_s) in salami_circles:
                    cv2.circle(combined_image, (int(center_s[0]), int(center_s[1])), 
                              int(radius_s), (0, 0, 255), 2)
                cv2.imwrite(str(debug_dir / f"combined_{image_path.name}"), combined_image)
            
            # Step 4: Get segmentation masks for analysis
            pizza_mask = pizza_seg_service.segment_pizza(working_image_path)
            salami_mask = salami_seg_service.segment_salami(working_image_path)
            
            # Save masks
            cv2.imwrite(str(debug_dir / f"pizza_mask_{image_path.name}.png"), pizza_mask)
            cv2.imwrite(str(debug_dir / f"salami_mask_{image_path.name}.png"), salami_mask)
            
            # Print statistics
            print(f"  Statistics:")
            print(f"    - Pizza area: {np.sum(pizza_mask == 255):,} pixels")
            print(f"    - Salami area: {np.sum(salami_mask == 255):,} pixels")
            print(f"    - Salami coverage: {np.sum(salami_mask == 255) / np.sum(pizza_mask == 255) * 100:.1f}%")
            
            print(f"  ✓ Results saved to debug/")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    print(f"Processing complete. Check the debug/ directory for results.")


if __name__ == "__main__":
    main()