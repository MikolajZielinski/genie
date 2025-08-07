#!/bin/bash
set -e

echo "🚀 Starting GENIE container..."

# Check if OptiX is already built
if [ ! -f /workspace/genie/genie/knn/optix_knn.so ]; then
    echo "🔧 Building OptiX with GPU detection..."
    cd /workspace/genie/genie/knn
    ./build_optix.sh
    echo "✅ OptiX build complete"
else
    echo "✅ OptiX already built, skipping..."
fi

echo "📦 Installing GENIE package..."
cd /workspace/genie
pip install -e .
ns-install-cli
echo "✅ GENIE installation complete"
echo "🎉 Container ready!"

# Execute the original command
exec "$@"
