# Pizza Split Project Documentation

## Project Overview
This project provides computer vision services for pizza analysis, including pizza segmentation, circle detection, and salami detection. The main goal is to detect and analyze pizza shapes and toppings (specifically salami) for portion splitting purposes.

## Project Structure

```
rocket-backend/
├── src/
│   ├── service/pizza_split/
│   │   ├── pizza_segmentation_service.py      # Pizza segmentation using YOLO
│   │   ├── pizza_circle_detection_service.py  # Pizza circle approximation
│   │   ├── salami_segmentation_service.py     # Salami color-based detection
│   │   ├── salami_circle_detection_service.py # Individual salami circle detection
│   │   └── preprocess.py                      # Image preprocessing (ellipse→circle)
│   └── test/
│       └── debug.py                           # Debug pipeline script
├── resource/                                  # Input images
│   ├── pizza1.jpg
│   └── pizza2.jpg
├── result/                                    # Processing results
├── debug/                                     # Debug outputs
├── venv/                                      # Python virtual environment
└── yolov8n-seg.pt                            # YOLO model weights
```

## Services Documentation

### 1. `pizza_segmentation_service.py`
**Purpose**: Segment pizza regions from images using deep learning

**Key Features**:
- Uses YOLOv8n segmentation model
- Detects pizza class (ID: 53)
- Returns binary mask (255=pizza, 0=background)

**Main Method**: `segment_pizza(image_path) -> np.ndarray`

### 2. `pizza_circle_detection_service.py`
**Purpose**: Approximate pizza shape as a circle

**Key Features**:
- Uses pizza segmentation mask
- Finds largest contour and fits minimum enclosing circle
- Returns center coordinates and radius
- Can overlay circle visualization on images

**Main Methods**:
- `detect_circle_from_image(image_path) -> ((center_x, center_y), radius)`
- `draw_circle_on_image(image, center, radius) -> np.ndarray`

### 3. `salami_segmentation_service.py`
**Purpose**: Detect salami regions using color-based analysis

**Key Features**:
- K-means color quantization (6 clusters)
- HSV color space filtering for red colors (salami)
- Bilateral filtering for edge preservation
- Morphological operations for noise reduction
- Minimum area threshold (1000 pixels)

**Color Thresholds**:
- Hue: 0-8° and 172-180° (red ranges)
- Saturation: ≥120
- Value (brightness): ≥90

**Main Method**: `segment_salami(image_path) -> np.ndarray`

### 4. `salami_circle_detection_service.py`
**Purpose**: Detect individual salami pieces as circles

**Key Features**:
- Combines pizza and salami masks (AND operation)
- Multi-stage morphological separation of connected salamis
- Circularity filtering (>0.3) and minimum radius (>12px)
- Minimum area threshold (200 pixels)

**Processing Pipeline**:
1. Get salami and pizza masks
2. Apply bitwise AND to get salami within pizza bounds
3. Erosion with ellipse kernel (11x11) for separation
4. Cross-shaped erosion (5x5) for stubborn connections
5. Dilation recovery (7x7 ellipse)
6. Small noise cleanup
7. Circle detection from contours

**Main Methods**:
- `detect_salami_circles(image_path) -> List[((center_x, center_y), radius)]`
- `draw_circles_on_image(image, circles) -> np.ndarray`

### 5. `preprocess.py`
**Purpose**: Preprocess images to convert elliptical pizzas to circular shape

**Key Features**:
- Ellipse detection from pizza segmentation
- Elliptical shape detection (>10% axis difference)
- Affine transformation to convert ellipse to circle
- Preserves image quality and proportions

**Transformation Process**:
1. Translate pizza center to origin
2. Rotate to align with coordinate axes
3. Scale major/minor axes to equal length
4. Rotate back to original orientation
5. Translate back to original position

**Main Method**: `preprocess_pizza_image(image_path, output_path) -> (preprocessed_image, info_dict)`

### 6. `debug.py`
**Purpose**: Complete pipeline testing and visualization

**Pipeline Steps**:
1. **Preprocessing**: Convert elliptical pizzas to circular
2. **Pizza Circle Detection**: Detect overall pizza boundary
3. **Salami Circle Detection**: Detect individual salami pieces
4. **Visualization**: Generate debug images

**Output Files**:
- `debug/pizza_circle.png`: Pizza circle detection result
- `debug/salami_circle.png`: Salami circles detection result
- `debug/preprocessed_*.jpg`: Preprocessed images
- `debug/combined_*.jpg`: Combined visualizations

## Setup and Execution

### Prerequisites
- Python 3.x
- Virtual environment with required packages

### Required Packages
```bash
pip install opencv-python ultralytics numpy pathlib
```

### Execution Steps

1. **Activate Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Run Debug Pipeline**:
   ```bash
   python3 src/test/debug.py
   ```

3. **Run Individual Services** (optional):
   ```bash
   # Pizza segmentation
   python3 src/service/pizza_split/pizza_segmentation_service.py
   
   # Pizza circle detection
   python3 src/service/pizza_split/pizza_circle_detection_service.py
   
   # Salami segmentation
   python3 src/service/pizza_split/salami_segmentation_service.py
   
   # Salami circle detection
   python3 src/service/pizza_split/salami_circle_detection_service.py
   
   # Preprocessing
   python3 src/service/pizza_split/preprocess.py
   ```

## Input/Output Specifications

### Input Images
- **Location**: `resource/` directory
- **Format**: JPG images
- **Expected Content**: Pizza images (can be elliptical)

### Output Results
- **Preprocessed Images**: `debug/preprocessed_*.jpg`
- **Pizza Circle Visualization**: `debug/pizza_circle.png`
- **Salami Circles Visualization**: `debug/salami_circle.png`
- **Combined Visualization**: `debug/combined_*.jpg`

## Magic Numbers and Thresholds

### Salami Detection
- `COLOR_QUANTIZATION_K = 6`: K-means clusters
- `SATURATION_THRESHOLD = 120`: Minimum saturation
- `VALUE_THRESHOLD = 90`: Minimum brightness
- `MIN_CONTOUR_AREA = 1000`: Minimum salami area (pixels)

### Circle Detection
- `MIN_CONTOUR_AREA = 200`: Minimum for salami circles
- `CIRCULARITY_THRESHOLD = 0.3`: Minimum circularity
- `MIN_RADIUS = 12`: Minimum salami radius (pixels)

### Preprocessing
- `ELLIPSE_THRESHOLD = 0.1`: 10% axis difference for ellipse detection

## Common Issues and Solutions

1. **No circles detected**: Check image quality and lighting
2. **Too many false positives**: Adjust color thresholds or circularity filters
3. **Missing salami**: Lower area thresholds or adjust color ranges
4. **Transformation artifacts**: Check ellipse detection parameters

## Development Notes

- All services are designed to work independently
- Use the debug script for comprehensive testing
- Adjust magic numbers based on your specific pizza images
- The preprocessing step is crucial for accurate circle detection