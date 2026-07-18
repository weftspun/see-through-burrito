# see-through-burrito

Elixir implementation of [See-Through](https://github.com/weftspun/see-through) (Shitagaki Lab, SIGGRAPH 2026):
decompose anime illustrations into up to 24 semantic layers using neural networks, with GPU acceleration via
ExLA and self-contained executable packaging via Burrito.

```
see-through-burrito: Anime illustration decomposition

Usage:
  see-through-burrito --input <image> [OPTIONS]

Options:
  -i, --input <path>           Input image path (required)
  -o, --output <path>          Output path (default: ./output.svg)
  --steps <n>                  Diffusion steps (default: 30)
  --guidance <scale>           Guidance scale (default: 7.5)
  --depth-steps <n>            Depth estimation steps (default: 4)
  --width <px>                 Target width (default: 1024)
  --height <px>                Target height (default: 1024)
  --seed <n>                   Random seed (default: 42)
  --model-cache <dir>          Model cache directory
  -h, --help                   Show this help
```

```bash
./see_through_burrito \
  --input anime.png \
  --output layers.svg \
  --steps 50 \
  --guidance 10.0
```

## Quick Start with GPU

### Automatic Setup (Recommended)

```bash
# One-command GPU configuration
bash setup_gpu.sh

# This will:
# 1. Detect your CUDA version
# 2. Set XLA_TARGET automatically
# 3. Verify GPU is visible
# 4. Compile with GPU support
# 5. Test GPU connectivity
# 6. Save configuration to .env.gpu

# Then load config for future sessions
source .env.gpu
mix test.gpu
```

### Manual Setup

```bash
# Check CUDA version
nvcc --version

# Set environment (choose based on CUDA version)
export XLA_TARGET=cuda12  # For CUDA 12.x
# OR
export XLA_TARGET=cuda13  # For CUDA 13.x

# Compile
mix compile

# Verify GPU setup
mix run -e "IO.inspect(EXLA.Client.list())"
# Should show [:cuda] not [:cpu]
```

**Important**: `XLA_TARGET` must be set BEFORE `mix compile`. See [GPU_TESTING.md](GPU_TESTING.md) for detailed setup.
