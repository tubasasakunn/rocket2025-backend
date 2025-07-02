FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file
COPY yolov8n-seg.pt .

# Copy the application code
COPY src/ src/
COPY api/ api/
COPY runpod_handler.py .

# Make upload directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "runpod_handler"]
