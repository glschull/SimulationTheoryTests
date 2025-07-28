# Dockerfile for Simulation Theory Test Kit
# Ensures reproducible scientific computing environment

FROM python:3.13.3-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data results

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV MPLBACKEND="Agg"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash simuser
RUN chown -R simuser:simuser /app
USER simuser

# Default command
CMD ["python", "main_runner.py", "--all"]

# Metadata
LABEL maintainer="Simulation Theory Test Kit"
LABEL description="Scientific framework for testing simulation hypothesis"
LABEL version="1.0"
LABEL created="2025-07-27"
