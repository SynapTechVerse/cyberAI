# Use the official NVIDIA CUDA base image with runtime support
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set environment variables to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies, including Python, pip, and other necessary libraries
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libjpeg-progs \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Install pip for Python 3 (if not already installed)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Create a symlink to make `python` point to `python3`
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python dependencies globally in one step
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
    opencv-python-headless \
    face_recognition \
    Pillow \
    dlib  \
    psutil  # Ensure CUDA support for dlib if available

# Create the output directory for result.txt and ensure write permissions
RUN mkdir -p /app/output && chmod 777 /app/output && chmod 777 /app
RUN mkdir -p /app/suspect/suspect_abscent && chmod 777 /app/suspect/suspect_abscent
RUN mkdir -p /app/suspect/suspect_persent && chmod 777 /app/suspect/suspect_persent

# Copy the app.py into the container
COPY app.py /app/app.py

# Ensure the app directory structure is in place
RUN mkdir -p /app/img/document/a /app/img/document/b

# Run the application directly when the container starts
CMD ["python", "app.py"]
