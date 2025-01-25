# Use the official NVIDIA CUDA base image with runtime support
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set environment variables to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
USER root
# Set working directory
WORKDIR /app
# Create the output directory for result.txt and ensure write permissions


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
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Install pip for Python 3 (if not already installed)
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3  # Not needed if python3-pip is already installed

# Create a symlink to make `python` point to `python3`
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python dependencies globally in one step
RUN pip install --upgrade pip \
    && pip install -v --no-cache-dir \
    opencv-python-headless \
    face_recognition \
    Pillow \
    dlib > /tmp/pip_install.log 2>&1 \
    && tail -n 20 /tmp/pip_install.log


RUN mkdir -p /app/output /app/output/suspect/suspect_persent && \
chmod -R 777 /app/output  /app/output/suspect/suspect_persent /app

RUN mkdir -p /app/output/suspect/suspect_abscent/ && \
chmod -R 777 /app/output/suspect/suspect_abscent/


# Copy the app.py into the container
COPY app.py /app/app.py
COPY startup.sh /app/startup.sh

# Ensure the app directory structure is in place
RUN mkdir -p /app/img/document/a /app/img/document/b 

# Make the startup script executable
RUN chmod +x /app/startup.sh

# Run the application directly when the container starts
#CMD ["python", "app.py"]
# Set the default command to run the startup script
CMD ["/app/startup.sh"]
