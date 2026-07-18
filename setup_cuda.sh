#!/bin/bash
# CUDA environment setup via pixi

echo "🎯 Setting up CUDA environment via pixi..."

# Set LD_LIBRARY_PATH to include pixi environment CUDA libraries
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# CUDA-specific paths
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$CONDA_PREFIX"
export CUDNN_PATH="$CONDA_PREFIX"

# Set XLA_TARGET for Elixir
export XLA_TARGET=cuda13

echo "✅ CUDA environment configured:"
echo "   CUDA_PATH: $CUDA_PATH"
echo "   LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "   XLA_TARGET: $XLA_TARGET"

# Verify CUDA is accessible
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA compiler (nvcc) found:"
    nvcc --version
else
    echo "⚠️  CUDA compiler (nvcc) not in PATH"
fi

# Check for NCCL
if [ -f "$CONDA_PREFIX/lib/libnccl.so.2" ]; then
    echo "✅ NCCL library found: $CONDA_PREFIX/lib/libnccl.so.2"
else
    echo "⚠️  NCCL library not found"
fi
