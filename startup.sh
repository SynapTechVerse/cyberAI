#!/bin/bash

# Directories to create inside the container (will be reflected on the host)
DIRS=(
  "/app/output/suspect/suspect_persent"
  "/app/output/suspect/suspect_abscent"
)

# Check and create directories only if they do not exist
for dir in "${DIRS[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Creating directory: $dir"
    mkdir -p $dir
    chmod -R 777 $dir
  else
    echo "Directory already exists: $dir"
  fi
done

# Start the Python application after directory creation
python /app/app.py
