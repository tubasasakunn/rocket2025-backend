#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# プロジェクトルートをPythonパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.service.pizza_split.pizza_segmentation_service import PizzaSegmentationService
from src.service.pizza_split.salami_segmentation_service import SalamiSegmentationService

def process_pizza_image(image_path, output_dir=None, isDebug=False):
    """
    Process pizza image through the complete pipeline:
    1. Pizza segmentation
    2. Salami segmentation
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output masks (optional)
        isDebug: Enable debug output
        
    Returns:
        dict: Contains original image, pizza mask, salami mask
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Initialize services
    pizza_service = PizzaSegmentationService()
    salami_service = SalamiSegmentationService()
    
    # Load original image
    original_image = cv2.imread(str(image_path))
    results['original_image'] = original_image
    
    # Step 1: Pizza segmentation
    print(f"Step 1: Segmenting pizza...")
    pizza_mask = pizza_service.segment_pizza(str(image_path), isDebug=isDebug)
    results['pizza_mask'] = pizza_mask
    print(f"  Pizza segmentation complete. Mask shape: {pizza_mask.shape}")
    
    # Step 2: Salami segmentation
    print(f"Step 2: Segmenting salami...")
    salami_result = salami_service.segment_salami(
        str(image_path), 
        pizza_mask=pizza_mask,
        debug_output_dir=output_dir if output_dir and isDebug else None,
        base_name=image_path.stem,
        isDebug=isDebug
    )
    if isinstance(salami_result, tuple):
        salami_mask, salami_debug_info = salami_result
        results['salami_debug_info'] = salami_debug_info
    else:
        salami_mask = salami_result
    results['salami_mask'] = salami_mask
    print(f"  Salami segmentation complete. Mask shape: {salami_mask.shape}")
    
    # Step 3: Analyze each pizza region separately
    print(f"\nStep 3: Analyzing each pizza region...")
    region_stats = analyze_pizza_regions(pizza_mask, salami_mask)
    results['region_stats'] = region_stats
    
    # Save masks if output directory specified
    if output_dir:
        # Save individual masks
        cv2.imwrite(str(output_dir / "pizza_mask.png"), pizza_mask)
        cv2.imwrite(str(output_dir / "salami_mask.png"), salami_mask)
        cv2.imwrite(str(output_dir / "original.jpg"), original_image)
        
        # Create overlay visualization
        overlay = create_overlay_visualization(
            original_image, 
            pizza_mask, 
            salami_mask
        )
        cv2.imwrite(str(output_dir / "overlay_result.png"), overlay)
        print(f"  Saved results to {output_dir}")
    
    return results

def create_overlay_visualization(image, pizza_mask, salami_mask):
    """
    Create an overlay visualization showing original image with colored masks
    
    Args:
        image: Original/preprocessed image
        pizza_mask: Binary mask for pizza
        salami_mask: Binary mask for salami
        
    Returns:
        Overlay image with colored masks
    """
    # Create colored overlay
    overlay = image.copy()
    
    # Create colored masks
    pizza_color = np.zeros_like(image)
    pizza_color[:, :, 1] = pizza_mask  # Green for pizza
    
    salami_color = np.zeros_like(image)
    salami_color[:, :, 2] = salami_mask  # Red for salami
    
    # Apply masks with transparency
    alpha = 0.3
    
    # Apply pizza mask if there are any pizza pixels
    pizza_mask_area = pizza_mask > 0
    if np.any(pizza_mask_area):
        pizza_overlay = cv2.addWeighted(
            overlay, 1-alpha, 
            pizza_color, alpha, 0
        )
        overlay[pizza_mask_area] = pizza_overlay[pizza_mask_area]
    
    # Apply salami mask if there are any salami pixels
    salami_mask_area = salami_mask > 0
    if np.any(salami_mask_area):
        salami_overlay = cv2.addWeighted(
            overlay, 1-alpha*2, 
            salami_color, alpha*2, 0
        )
        overlay[salami_mask_area] = salami_overlay[salami_mask_area]
    
    # Add legend
    legend_height = 60
    legend = np.ones((legend_height, overlay.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(legend, "Pizza (Green)", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(legend, "Salami (Red)", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    
    # Combine image and legend
    result = np.vstack([overlay, legend])
    
    return result

def calculate_scores(pizza_mask, salami_mask):
    """
    Calculate various scores from the masks
    
    Args:
        pizza_mask: Binary mask for pizza
        salami_mask: Binary mask for salami
        
    Returns:
        dict: Various calculated scores
    """
    scores = {}
    
    # Calculate areas
    pizza_area = np.sum(pizza_mask > 0)
    salami_area = np.sum(salami_mask > 0)
    
    scores['pizza_area_pixels'] = pizza_area
    scores['salami_area_pixels'] = salami_area
    scores['salami_coverage_ratio'] = salami_area / pizza_area if pizza_area > 0 else 0
    scores['salami_coverage_percent'] = scores['salami_coverage_ratio'] * 100
    
    # Calculate salami distribution (simple quadrant analysis)
    h, w = pizza_mask.shape
    mid_h, mid_w = h // 2, w // 2
    
    quadrants = {
        'top_left': salami_mask[:mid_h, :mid_w],
        'top_right': salami_mask[:mid_h, mid_w:],
        'bottom_left': salami_mask[mid_h:, :mid_w],
        'bottom_right': salami_mask[mid_h:, mid_w:]
    }
    
    quadrant_scores = {}
    for name, quadrant in quadrants.items():
        quadrant_scores[name] = np.sum(quadrant > 0)
    
    scores['quadrant_distribution'] = quadrant_scores
    
    # Calculate distribution uniformity (standard deviation of quadrant scores)
    quadrant_values = list(quadrant_scores.values())
    scores['distribution_std'] = np.std(quadrant_values)
    scores['distribution_uniformity'] = 1 - (scores['distribution_std'] / np.mean(quadrant_values)) if np.mean(quadrant_values) > 0 else 0
    
    return scores

def calculate_fairness_score(pizza_pixels_list, salami_pixels_list, pizza_weight=0.3, salami_weight=0.7):
    """
    Calculate fairness score based on standard deviation of pizza and salami distribution
    
    Args:
        pizza_pixels_list: List of pizza pixels per region
        salami_pixels_list: List of salami pixels per region
        pizza_weight: Weight for pizza fairness (default 0.3)
        salami_weight: Weight for salami fairness (default 0.7)
        
    Returns:
        Fairness score from 0 to 100 (100 = perfectly fair, 0 = very unfair)
    """
    if len(pizza_pixels_list) <= 1:
        return 100.0  # Single region is perfectly fair by definition
    
    # Calculate coefficient of variation (CV) for normalization
    # CV = std / mean, which is scale-independent
    mean_pizza = np.mean(pizza_pixels_list)
    mean_salami = np.mean(salami_pixels_list)
    
    cv_pizza = np.std(pizza_pixels_list) / mean_pizza if mean_pizza > 0 else 0
    cv_salami = np.std(salami_pixels_list) / mean_salami if mean_salami > 0 else 0
    
    # Convert CV to fairness score using exponential decay
    # Score = 100 * exp(-k * CV), where k controls the decay rate
    k = 3.0  # Decay rate (adjust to control sensitivity)
    
    pizza_fairness = 100 * np.exp(-k * cv_pizza)
    salami_fairness = 100 * np.exp(-k * cv_salami)
    
    # Weighted average
    total_weight = pizza_weight + salami_weight
    fairness_score = (pizza_weight * pizza_fairness + salami_weight * salami_fairness) / total_weight
    
    return fairness_score

def analyze_pizza_regions(pizza_mask, salami_mask):
    """
    Analyze each connected pizza region separately
    
    Args:
        pizza_mask: Binary mask for pizza regions
        salami_mask: Binary mask for salami regions
        
    Returns:
        Dictionary containing:
        - regions: List of stats for each region
        - std_pizza: Standard deviation of pizza pixels across regions
        - std_salami: Standard deviation of salami pixels across regions
    """
    # Find connected components in pizza mask
    num_labels, labels = cv2.connectedComponents(pizza_mask.astype(np.uint8))
    
    region_stats = []
    pizza_pixels_list = []
    salami_pixels_list = []
    
    print(f"  Found {num_labels - 1} pizza regions")
    
    for label in range(1, num_labels):  # Skip background (label 0)
        # Create mask for current region
        region_mask = (labels == label).astype(np.uint8) * 255
        
        # Calculate pizza pixels in this region
        pizza_pixels = np.sum(region_mask > 0)
        
        # Calculate salami pixels in this region (intersection)
        salami_in_region = cv2.bitwise_and(region_mask, salami_mask)
        salami_pixels = np.sum(salami_in_region > 0)
        
        # Calculate coverage
        coverage = salami_pixels / pizza_pixels if pizza_pixels > 0 else 0
        
        stats = {
            'region_id': label,
            'pizza_pixels': pizza_pixels,
            'salami_pixels': salami_pixels,
            'coverage_ratio': coverage,
            'coverage_percent': coverage * 100
        }
        
        region_stats.append(stats)
        pizza_pixels_list.append(pizza_pixels)
        salami_pixels_list.append(salami_pixels)
        
        print(f"  Region {label}:")
        print(f"    - Pizza pixels: {pizza_pixels:,}")
        print(f"    - Salami pixels: {salami_pixels:,}")
        print(f"    - Coverage: {coverage * 100:.1f}%")
    
    # Calculate standard deviations
    std_pizza = np.std(pizza_pixels_list) if len(pizza_pixels_list) > 1 else 0
    std_salami = np.std(salami_pixels_list) if len(salami_pixels_list) > 1 else 0
    
    print(f"\n  Standard deviations across regions:")
    print(f"    - Pizza pixels std: {std_pizza:,.1f}")
    print(f"    - Salami pixels std: {std_salami:,.1f}")
    
    # Calculate fairness score
    fairness_score = calculate_fairness_score(
        pizza_pixels_list, salami_pixels_list, 
        pizza_weight=0.3, salami_weight=0.7
    )
    
    print(f"\n  Fairness Score: {fairness_score:.1f} / 100")
    
    return {
        'regions': region_stats,
        'std_pizza': std_pizza,
        'std_salami': std_salami,
        'pizza_pixels_list': pizza_pixels_list,
        'salami_pixels_list': salami_pixels_list,
        'fairness_score': fairness_score
    }

if __name__ == "__main__":
    # Process cutted1.jpg with morphological separation
    image_path = "resource/cutted1.jpg"
    output_dir = "result/score_output_cutted1_morph"
    
    print(f"Processing {image_path}...")
    print("=" * 50)
    
    try:
        # Process the image
        results = process_pizza_image(
            image_path, 
            output_dir=output_dir,
            isDebug=True
        )
        
        # Calculate scores
        scores = calculate_scores(
            results['pizza_mask'], 
            results['salami_mask']
        )
        
        print("\n" + "=" * 50)
        print("SCORING RESULTS:")
        print("=" * 50)
        print(f"Pizza area: {scores['pizza_area_pixels']:,} pixels")
        print(f"Salami area: {scores['salami_area_pixels']:,} pixels")
        print(f"Salami coverage: {scores['salami_coverage_percent']:.1f}%")
        print(f"\nQuadrant distribution:")
        for quadrant, value in scores['quadrant_distribution'].items():
            print(f"  {quadrant}: {value:,} pixels")
        print(f"\nDistribution uniformity: {scores['distribution_uniformity']:.2f} (0=uneven, 1=perfect)")
        
        # Print per-region statistics
        print("\n" + "=" * 50)
        print("PER-REGION STATISTICS:")
        print("=" * 50)
        for region in results['region_stats']['regions']:
            print(f"Region {region['region_id']}:")
            print(f"  Pizza area: {region['pizza_pixels']:,} pixels")
            print(f"  Salami area: {region['salami_pixels']:,} pixels")
            print(f"  Salami coverage: {region['coverage_percent']:.1f}%")
        
        print(f"\nREGION BALANCE METRICS:")
        print(f"  Pizza area std deviation: {results['region_stats']['std_pizza']:,.1f} pixels")
        print(f"  Salami area std deviation: {results['region_stats']['std_salami']:,.1f} pixels")
        print(f"  FAIRNESS SCORE: {results['region_stats']['fairness_score']:.1f} / 100")
        print(f"    (Pizza weight: 30%, Salami weight: 70%)")
        
        print(f"\nAll results saved to: {output_dir}")
        print("  - pizza_mask.png: Pizza segmentation mask")
        print("  - salami_mask.png: Salami segmentation mask")
        print("  - original.jpg: Original image")
        print("  - overlay_result.png: Overlay visualization")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()