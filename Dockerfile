FROM python:3.10-slim

# System dependencies for OpenCV and DeepFace
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ArcFace model at build time
RUN python -c "\
from deepface import DeepFace; \
import numpy as np; \
dummy = np.zeros((112,112,3), dtype='uint8'); \
DeepFace.represent(img_path=dummy, model_name='ArcFace', detector_backend='opencv', enforce_detection=False); \
print('ArcFace model downloaded.')"

# Copy app files
COPY face_auth_backend.py .
COPY face_auth_frontend.html .

# Create data dirs
RUN mkdir -p /app/data/dataset

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Run on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "face_auth_backend:app"]
