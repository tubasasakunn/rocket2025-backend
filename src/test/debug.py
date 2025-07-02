import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from service.pizza_split.preprocess import PreprocessService
from service.pizza_split.pizza_circle_detection_service import PizzaCircleDetectionService
from service.pizza_split.salami_circle_detection_service import SalamiCircleDetectionService


def create_debug_directories():
    """Create debug output directories"""
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    return debug_dir


def run_preprocessing_pipeline(input_image_path: str, debug_dir: Path):
    """
    Run the complete preprocessing and detection pipeline
    
    Args:
        input_image_path: Path to input image
        debug_dir: Directory for debug outputs
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_image_path}")
    print(f"{'='*60}")
    
    # Step 1: Preprocess image
    print("\n1. Preprocessing image...")
    preprocess_service = PreprocessService()
    
    # Save preprocessed image
    preprocessed_path = debug_dir / f"preprocessed_{Path(input_image_path).name}"
    try:
        preprocessed_image, preprocess_info = preprocess_service.preprocess_pizza_image(
            input_image_path, 
            str(preprocessed_path)
        )
        
        if preprocess_info['is_transformed']:
            print("   ✓ Image was transformed from ellipse to circle")
        else:
            print("   ✓ Image was already circular")
    except Exception as e:
        print(f"   ✗ Preprocessing failed: {e}")
        return
    
    # Step 2: Detect pizza circle
    print("\n2. Detecting pizza circle...")
    pizza_service = PizzaCircleDetectionService()
    
    try:
        # Detect circle from preprocessed image
        circle_info = pizza_service.detect_circle_from_image(str(preprocessed_path))
        
        if circle_info:
            center, radius = circle_info
            print(f"   ✓ Pizza circle detected: Center={center}, Radius={radius}")
            
            # Draw and save pizza circle
            result_image = pizza_service.draw_circle_on_image(preprocessed_image, center, radius)
            pizza_circle_path = debug_dir / "pizza_circle.png"
            cv2.imwrite(str(pizza_circle_path), result_image)
            print(f"   ✓ Saved pizza circle visualization to {pizza_circle_path}")
        else:
            print("   ✗ No pizza circle detected")
            return
    except Exception as e:
        print(f"   ✗ Pizza circle detection failed: {e}")
        return
    
    # Step 3: Detect salami circles
    print("\n3. Detecting salami circles...")
    salami_service = SalamiCircleDetectionService()
    
    try:
        # Detect salami circles from preprocessed image
        salami_circles = salami_service.detect_salami_circles(str(preprocessed_path))
        
        print(f"   ✓ Detected {len(salami_circles)} salami circles")
        
        if salami_circles:
            # Draw and save salami circles
            salami_result = salami_service.draw_circles_on_image(preprocessed_image, salami_circles)
            salami_circle_path = debug_dir / "salami_circle.png"
            cv2.imwrite(str(salami_circle_path), salami_result)
            print(f"   ✓ Saved salami circles visualization to {salami_circle_path}")
            
            # Print details
            for i, ((cx, cy), r) in enumerate(salami_circles):
                print(f"     - Salami {i+1}: Center=({cx}, {cy}), Radius={r}")
        else:
            print("   ⚠ No salami circles detected")
    except Exception as e:
        print(f"   ✗ Salami circle detection failed: {e}")
        return
    
    # Step 4: Create combined visualization
    print("\n4. Creating combined visualization...")
    try:
        # Start with preprocessed image
        combined_image = preprocessed_image.copy()
        
        # Draw pizza circle in green
        if circle_info:
            center, radius = circle_info
            cv2.circle(combined_image, center, radius, (0, 255, 0), 3)
            cv2.circle(combined_image, center, 5, (0, 255, 0), -1)
        
        # Draw salami circles in red
        for (cx, cy), r in salami_circles:
            cv2.circle(combined_image, (cx, cy), r, (0, 0, 255), 2)
            cv2.circle(combined_image, (cx, cy), 3, (0, 0, 255), -1)
        
        # Add text summary
        text = f"Pizza: 1 circle, Salami: {len(salami_circles)} circles"
        cv2.putText(combined_image, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        combined_path = debug_dir / f"combined_{Path(input_image_path).name}"
        cv2.imwrite(str(combined_path), combined_image)
        print(f"   ✓ Saved combined visualization to {combined_path}")
    except Exception as e:
        print(f"   ✗ Combined visualization failed: {e}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")


def main():
    """Main debug function"""
    # Create debug directories
    debug_dir = create_debug_directories()
    print(f"Debug output directory: {debug_dir}")
    
    # List of test images
    test_images = [
        "resource/pizza.jpg",
        "resource/pizza1.jpg",
        "resource/pizza2.jpg"
    ]
    
    # Process each image
    for image_path in test_images:
        if Path(image_path).exists():
            run_preprocessing_pipeline(image_path, debug_dir)
        else:
            print(f"\n⚠ Skipping {image_path} - file not found")
    
    print("\n✅ All processing complete!")
    print(f"Check the '{debug_dir}' directory for results:")
    print("  - pizza_circle.png: Pizza circle detection result")
    print("  - salami_circle.png: Salami circles detection result")
    print("  - preprocessed_*.jpg: Preprocessed images")
    print("  - combined_*.jpg: Combined visualizations")


if __name__ == "__main__":
    main()