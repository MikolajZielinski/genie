# ========================
# Build Stage - Prepare environment without GPU-dependent compilation
# ========================
FROM ghcr.io/nerfstudio-project/nerfstudio:latest AS builder

# Set working directory
WORKDIR /workspace

# Install build dependencies (including CUDA toolkit for compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    python3-pip \
    git \
    ninja-build \
    cuda-toolkit=12.6.0-1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python build dependencies
RUN pip install --no-cache-dir \
    pybind11

# Install Blender
RUN wget https://download.blender.org/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz && \
    tar xvf blender-4.4.3-linux-x64.tar.xz && \
    rm blender-4.4.3-linux-x64.tar.xz

# ========================
# Runtime Stage - Runtime compilation with GPU access
# ========================
FROM ghcr.io/nerfstudio-project/nerfstudio:latest

# Set working directory
WORKDIR /workspace

# Install runtime dependencies (including CUDA toolkit for OptiX compilation at runtime)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    python3-pip \
    git \
    ninja-build \
    cuda-toolkit=12.6.0-1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
RUN pip install --no-cache-dir \
    pybind11

# Copy Blender from build stage
COPY --from=builder /workspace/blender-4.4.3-linux-x64 /workspace/blender-4.4.3-linux-x64
RUN ln -s /workspace/blender-4.4.3-linux-x64/blender /usr/local/bin/blender

# Copy project files
COPY . /workspace/genie/

# Copy and set up entrypoint script
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Set entrypoint and default command
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]