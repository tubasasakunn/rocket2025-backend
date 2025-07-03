#!/bin/bash
# filepath: /Users/ryosukeanzai/src/github.com/tubasasakunn/rocket2025-backend/set_env.sh
# Script to set environment variables for development or testing

# Set a specific directory for YOLO configuration
export YOLO_CONFIG_DIR=/app/config/ultralytics

# Create the directory if it doesn't exist
mkdir -p $YOLO_CONFIG_DIR
chmod -R 777 $YOLO_CONFIG_DIR

echo "Environment variables set successfully"
echo "YOLO_CONFIG_DIR=$YOLO_CONFIG_DIR"

# Optionally pre-generate matplotlib font cache
python -c "import matplotlib.pyplot as plt; plt.figure(); plt.close()"
echo "Matplotlib font cache generated"
