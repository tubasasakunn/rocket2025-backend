FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install core dependencies first
RUN pip install --no-cache-dir pip setuptools wheel --upgrade && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir svgwrite==1.4.3 matplotlib==3.7.1

# Verify that svgwrite is installed
RUN python -c "import svgwrite; print(f'svgwrite version: {svgwrite.__version__}')"

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

# Debug: Print Python path and installed packages at startup
RUN pip freeze > /app/installed_packages.txt

# Default command
CMD ["python", "runpod_handler.py"]
