#!/bin/bash

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🐳 Building Docker image..."
# Build the Docker image from the project root
docker build -t genie:latest -f docker/Dockerfile "$PROJECT_ROOT"

# Determine user
if [ -z "$SUDO_USER" ]; then
    user=$USER
else
    user=$SUDO_USER
fi

echo "🚀 Running container for user: $user"

# Ensure XAUTH is set for X11 forwarding
export XAUTH=${XAUTH:-$HOME/.Xauthority}

# Allow container to use the host X11 server
xhost +local:docker

echo "🏃 Starting container..."
# Run Docker container
docker run --gpus all \
    --name=${user}_genie \
    --env="QT_X11_NO_MITSHM=1" \
    --env="DISPLAY=$DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PROJECT_ROOT:/workspace/genie" \
    --volume="$HOME/.cache:/root/.cache" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --privileged \
    --network=host \
    -it \
    --shm-size=64gb \
    genie:latest \
    bash
