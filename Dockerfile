FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy YOLOv8 model
COPY yolov8n-seg.pt .

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads/process

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
ENV RUNPOD_MODE=web
ENV PYTHONPATH=/app

# Expose the port for Render
EXPOSE 10000

# Run the application with Gunicorn + Uvicorn workers (recommended for production)
RUN pip install gunicorn

# Use Gunicorn with Uvicorn workers for better performance and stability
CMD gunicorn src.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 4 --timeout 120
