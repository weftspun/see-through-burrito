#!/bin/bash
# GPU Setup Script for see-through-burrito
# Automatically configures XLA_TARGET and verifies GPU setup

set -e

echo "🚀 see-through-burrito GPU Setup"
echo "================================="
echo ""

# Step 1: Check CUDA installation
echo "📋 Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA toolkit not found (nvcc command not found)"
    echo "   Please install CUDA from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "Cuda compilation" | grep -oE "[0-9]+\.[0-9]+" | head -1)
echo "✅ CUDA version: $CUDA_VERSION"

# Step 2: Determine XLA_TARGET
echo ""
echo "🎯 Determining XLA_TARGET..."
MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

if [ "$MAJOR" -eq 12 ]; then
    XLA_TARGET="cuda12"
    echo "✅ CUDA 12.x detected → XLA_TARGET=cuda12"
elif [ "$MAJOR" -eq 13 ]; then
    XLA_TARGET="cuda13"
    echo "✅ CUDA 13.x detected → XLA_TARGET=cuda13"
else
    echo "⚠️  CUDA $CUDA_VERSION may not be fully supported"
    echo "   Supported versions: 12.x, 13.x"
    read -p "   Enter XLA_TARGET (cuda12/cuda13): " XLA_TARGET
fi

# Step 3: Check GPU visibility
echo ""
echo "🖥️  Checking GPU visibility..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    echo "   This usually means NVIDIA drivers are not installed"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "❌ No NVIDIA GPUs detected"
    echo "   Check with: nvidia-smi"
    exit 1
fi

echo "✅ Found $GPU_COUNT GPU(s):"
nvidia-smi --list-gpus | sed 's/^/   /'

# Step 4: Export XLA_TARGET
echo ""
echo "📝 Configuring environment..."
export XLA_TARGET=$XLA_TARGET
echo "✅ Set XLA_TARGET=$XLA_TARGET"

# Step 5: Clean dependencies
echo ""
echo "🧹 Cleaning dependencies..."
mix deps.clean --all > /dev/null 2>&1 || true

# Step 6: Compile with XLA
echo ""
echo "🔨 Compiling with XLA GPU support..."
echo "   (This may take 2-5 minutes on first run as XLA is downloaded)"
export XLA_TARGET=$XLA_TARGET
mix compile --force

# Step 7: Verify GPU setup
echo ""
echo "✅ Verifying GPU setup..."
mix run -e "
IO.puts('')
IO.puts('EXLA Client Verification:')
clients = EXLA.Client.list()
IO.inspect(clients, label: 'Available clients')

if Enum.member?(clients, :cuda) or Enum.member?(clients, :gpu) do
  IO.puts('✅ GPU client available!')
  try do
    mem = EXLA.Client.memory_info(:default)
    IO.inspect(mem, label: 'GPU memory')
  rescue
    _ -> IO.puts('   (GPU memory info not available)')
  end
else
  IO.puts('⚠️  WARNING: GPU client not found, defaulting to CPU')
  IO.puts('   Run: export XLA_TARGET=$XLA_TARGET && mix compile --force')
end
"

# Step 8: Test with tensor operation
echo ""
echo "🧪 Running quick tensor test..."
mix run -e "
start_time = System.monotonic_time()
t = Nx.iota({1000, 1000}, type: :f32) |> Nx.sum()
end_time = System.monotonic_time()
elapsed_ms = (end_time - start_time) / 1_000_000
IO.puts(\"Tensor operation completed in #{Float.round(elapsed_ms, 2)}ms\")
if elapsed_ms < 50 do
  IO.puts('✅ GPU inference speed detected!')
else
  IO.puts('⚠️  Slow performance - may be running on CPU')
end
"

# Step 9: Save configuration
echo ""
echo "💾 Saving configuration..."
cat > .env.gpu << EOF
export XLA_TARGET=$XLA_TARGET
export CUDA_PATH=/usr/local/cuda
export PATH=\$CUDA_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH
EOF

echo "✅ Configuration saved to .env.gpu"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ GPU Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "To use GPU in future sessions:"
echo "  1. Load config: source .env.gpu"
echo "  2. Run tests: mix test.gpu"
echo ""
echo "Quick test commands:"
echo "  mix test test/exla_sanity_test.exs --exclude skip"
echo "  mix test.gpu test/bumblebee_api_test.exs"
echo "  mix test.gpu test/pipeline_e2e_test.exs"
echo ""
