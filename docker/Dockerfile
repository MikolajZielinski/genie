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

# Install Blender
RUN wget https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz
RUN tar xvf blender-4.4.3-linux-x64.tar.xz
RUN ln -s /workspace/blender-4.4.3-linux-x64/blender /usr/local/bin/blender

# Copy your project files
COPY . /workspace/genie/

# Build OptiX
WORKDIR /workspace/genie/genie/knn
RUN ./build_optix.sh

# Register GENIE in Nerfstudio
WORKDIR /workspace/genie
RUN pip install -e .
RUN ns-install-cli

# Default command
CMD ["bash"]