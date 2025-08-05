FROM ghcr.io/nerfstudio-project/nerfstudio:latest

# Set working directory
WORKDIR /workspace

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    python3-pip \
    git \
    ninja-build \
    cuda-toolkit=12.6.0-1 \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install --no-cache-dir \
    pybind11 

# Copy your project files
COPY . /workspace/genie/

# Set working directory to your project
WORKDIR /workspace/genie

# Default command
CMD ["bash"]