# ========================
# Base Image
# ========================
FROM python:3.10-slim

# ========================
# System Dependencies
# ========================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ========================
# Working Directory
# ========================
WORKDIR /app

# ========================
# Python Dependencies
# ========================
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    torch \
    torchvision \
    opencv-python-headless \
    pillow \
    numpy \
    python-multipart

# ========================
# Copy App Files
# ========================
COPY main.py .
COPY best_model.pth .

# ========================
# Expose Port
# ========================
EXPOSE 8000

# ========================
# Run App
# ========================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
