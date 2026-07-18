# GPU Setup Required - see-through-burrito

**Status**: 🔴 **GPU MANDATORY** - This project will NOT compile or run without GPU.

## Quick Start (60 seconds)

```bash
# 1. Install CUDA (if not already installed)
# 2. Run setup script
bash setup_gpu.sh

# 3. Compile
export XLA_TARGET=cuda13  # or cuda12 for CUDA 12.x
mix compile

# 4. Test
mix test test/exla_sanity_test.exs
```

---

## System Requirements

### Hardware
- **NVIDIA GPU** with CUDA Compute Capability 3.5+
- **Recommended**: RTX 4090, A100, or similar (24GB+ VRAM for full pipeline)
- **Minimum**: RTX 3070 or RTX 4060 (8GB+ VRAM)

### Software

#### Option A: CUDA Toolkit (Recommended)
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit nvidia-utils

# Or download from NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

#### Option B: Docker
```bash
docker run --gpus all -it nvidia/cuda:13.1-runtime-ubuntu22.04
cd /workspace
bash setup_gpu.sh
```

#### Option C: Conda/Pixi (Easiest for WSL2)
```bash
pixi env create --name cuda-elixir -c conda-forge -c nvidia \
  cuda-version=13.1 cuda-runtime=13.1

pixi run bash setup_gpu.sh
```

---

## Setup Steps

### 1. Verify CUDA Installation

```bash
# Check CUDA is installed
nvcc --version

# Check GPU is visible
nvidia-smi

# Example output:
# GPU 0: NVIDIA RTX 4090 (CUDA Capability 8.9)
```

### 2. Determine XLA_TARGET

| CUDA Version | XLA_TARGET |
|--------------|-----------|
| CUDA 13.x | `cuda13` |
| CUDA 12.x | `cuda12` |
| Apple Silicon | `metal` |

### 3. Set Environment

```bash
# Temporarily (for this session only)
export XLA_TARGET=cuda13

# Permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export XLA_TARGET=cuda13' >> ~/.bashrc
source ~/.bashrc
```

### 4. Run Automated Setup

```bash
bash setup_gpu.sh
```

This will:
- ✅ Detect CUDA version
- ✅ Auto-configure XLA_TARGET
- ✅ Verify GPU visibility
- ✅ Compile with GPU support
- ✅ Run verification tests
- ✅ Save configuration to `.env.gpu`

### 5. Compile Project

```bash
mix clean
mix compile --force
```

**First compile takes 2-5 minutes as XLA is downloaded.**

### 6. Verify GPU

```bash
mix run -e "IO.inspect(EXLA.Client.list())"
# Should show: [:cuda] or [:metal]

# Or run quick test
mix test test/exla_sanity_test.exs
```

---

## Troubleshooting

### ❌ "CUDA toolkit not found (nvcc command not found)"

**Solution:**
```bash
# Install CUDA
sudo apt-get install nvidia-cuda-toolkit

# Or add to PATH if already installed
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### ❌ "No NVIDIA GPUs detected"

**Solutions:**
1. Check NVIDIA drivers:
   ```bash
   lspci | grep -i nvidia
   nvidia-smi
   ```

2. Install NVIDIA drivers:
   ```bash
   sudo apt-get install nvidia-driver-535  # or latest version
   ```

3. In WSL2, enable GPU passthrough in `.wslconfig`:
   ```ini
   [wsl2]
   gpu=true
   ```

### ❌ "XLA_TARGET not set" at compile time

**Solution:**
```bash
export XLA_TARGET=cuda13
mix compile --force
```

### ❌ "CUDA library errors: NCCL not found"

**Solution (rebuild XLA from source):**
```bash
export XLA_BUILD=true
export XLA_TARGET=cuda13
mix clean
mix compile --force
```

**Or use conda:**
```bash
pixi run bash setup_gpu.sh
```

### ⚠️ Slow performance (>100ms for small tensors)

**Check if running on CPU:**
```bash
mix run -e "
t = Nx.iota({1000, 1000}, type: :f32) |> Nx.sum()
IO.inspect(t)
"
```

If takes >50ms, likely on CPU. Verify:
```bash
mix run -e "IO.inspect(EXLA.Client.list())"
mix run -e "IO.inspect(Nx.default_backend())"
```

Should show `EXLA.Backend`, not `Nx.BinaryBackend`.

---

## WSL2 Specific Setup

### Prerequisites

Enable GPU in WSL2 config:

```powershell
# In Windows PowerShell (as Admin)
cat > $env:USERPROFILE\.wslconfig << 'EOF'
[wsl2]
gpu=true
memory=16GB
processors=8
EOF
```

Then restart WSL:
```bash
wsl --shutdown
```

### Using Conda (Recommended for WSL2)

```bash
# Install pixi or conda
curl -fsSL https://pixi.sh/install.sh | bash

# Create environment
pixi env create --name see-through -c conda-forge -c nvidia \
  cuda-version=13.1 \
  cuda-runtime=13.1 \
  cudnn=9.1 \
  nccl=2.18

# Activate
pixi shell see-through

# Run setup
bash setup_gpu.sh
```

---

## Verification Checklist

After setup, verify each step:

```bash
# 1. CUDA installed
nvcc --version
# Expected: Cuda compilation tools, release 13.x

# 2. GPU visible
nvidia-smi
# Expected: Shows GPU name, memory, driver version

# 3. XLA_TARGET set
echo $XLA_TARGET
# Expected: cuda13 or cuda12

# 4. Project compiles
mix compile
# Expected: No errors, possibly some warnings

# 5. EXLA client available
mix run -e "IO.inspect(EXLA.Client.list())"
# Expected: [:cuda] or [:metal]

# 6. Tests pass (basic)
mix test test/exla_sanity_test.exs
# Expected: 4 passed

# 7. Performance good
mix run -e "t = Nx.iota({1000, 1000}) |> Nx.sum(); IO.inspect(t)"
# Expected: <50ms execution time
```

---

## Project-Specific Configuration

### Development (with GPU)

```bash
source .env.gpu
mix test
```

### Production Deployment

For Burrito releases:
```bash
bash setup_gpu.sh  # Once per machine
mix burrito.build
./burrito-output/see_through_burrito_linux -i input.png -o output.svg
```

---

## Performance Notes

### Expected Timings (RTX 4090)
- Tensor ops (1000x1000): <50ms
- Layer decomposition (1024x1024): ~2-5 seconds
- Full pipeline: ~15-30 seconds

### Memory Usage
- Base model: ~4GB VRAM
- Full pipeline: ~16-24GB VRAM recommended

### Batch Processing
To process multiple images efficiently:
```bash
for img in images/*.png; do
  ./see_through_burrito -i "$img" -o "output/$(basename $img .png).svg"
done
```

---

## References

- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- [ExLA GitHub](https://github.com/elixir-nx/elixir_nx)
- [Nx Documentation](https://hexdocs.pm/nx)
- [WSL2 GPU Support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

---

## Next Steps

Once GPU is configured:

1. **Run API discovery tests** to validate Bumblebee:
   ```bash
   mix test test/bumblebee_api_test.exs
   ```

2. **Run full test suite**:
   ```bash
   mix test
   ```

3. **Test with sample image**:
   ```bash
   mix escript.build
   ./see_through_burrito -i sample.png -o output.svg
   ```

---

**⚠️ This project WILL NOT work without GPU. No CPU fallback.**

For help: See `BLOCKERS.md` for known issues or `GPU_TESTING.md` for detailed testing guide.
