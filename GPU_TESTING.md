# GPU Testing Guide

## Quick Start (5 minutes)

```bash
# 1. Check CUDA version
nvcc --version
# Look for "Cuda compilation tools, release X.Y"

# 2. Set XLA_TARGET (CRITICAL!)
export XLA_TARGET=cuda12  # For CUDA 12.x
# OR
export XLA_TARGET=cuda13  # For CUDA 13.x

# 3. Verify GPU is visible
nvidia-smi

# 4. Clean and compile
mix deps.clean --all
mix compile

# 5. Test it works
mix test test/exla_sanity_test.exs --exclude skip
# Should pass with GPU results

# 6. Run full GPU tests
mix test.gpu test/bumblebee_api_test.exs
# Will download ~10-20GB of models on first run
```

**⚠️ Most common issue**: Forgetting step 2 (XLA_TARGET). Without it, ExLA runs on CPU and tests will be slow.

## Prerequisites

1. **NVIDIA GPU** with CUDA compute capability 5.0+
   - Minimum: GTX 750 (2GB VRAM for basic inference)
   - Recommended: RTX 3060+ (12GB+ VRAM for real workloads)
   - Required: 24GB+ VRAM for dual-pass LayerDiff + Marigold

2. **CUDA Toolkit** 11.0+ and **cuDNN** 8.0+
   ```bash
   # Verify CUDA installation
   nvcc --version
   
   # Verify cuDNN (should show multiple versions)
   ls /usr/local/cuda/lib64/libcudnn*
   
   # Expected CUDA versions:
   # - CUDA 12.x with cuDNN 9.8-9.11
   # - CUDA 13.x with cuDNN 9.12+
   ```

3. **ExLA with XLA_TARGET** (CRITICAL - must be set before mix compile)
   ```bash
   # Determine your CUDA version
   nvcc --version
   # Output: Cuda compilation tools, release X.Y
   
   # Set XLA_TARGET based on CUDA version
   export XLA_TARGET=cuda12  # For CUDA 12.x
   # OR
   export XLA_TARGET=cuda13  # For CUDA 13.x
   
   # Verify XLA can find CUDA
   mix compile
   # Should download/build XLA with CUDA support
   ```

4. **Elixir & Mix**
   ```bash
   elixir --version  # Should be 1.15+
   mix --version
   ```

## Environment Setup

### Critical: Set XLA_TARGET First

```bash
# BEFORE running any mix commands, determine CUDA version
nvcc --version

# Then export XLA_TARGET (choose one based on output above)
export XLA_TARGET=cuda12  # CUDA 12.1-12.x
# OR
export XLA_TARGET=cuda13  # CUDA 13.0+

# RECOMMENDED: Add to ~/.bashrc or ~/.zshrc to persist across sessions
echo 'export XLA_TARGET=cuda12' >> ~/.bashrc
source ~/.bashrc

# Verify
echo $XLA_TARGET
```

**Why this matters**: 
- Without `XLA_TARGET`, ExLA defaults to CPU-only
- Must be set BEFORE `mix compile` 
- Controls which XLA binary is downloaded/built
- Mismatched CUDA version causes runtime errors

### Linux/WSL2

```bash
# Step 1: Set XLA_TARGET (see above)

# Step 2: Set CUDA paths (if not in default location)
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Step 3: Verify CUDA is accessible
nvcc --version
nvidia-smi

# Step 4: Clean and recompile
mix deps.clean --all
mix compile
```

**Expected output**: Should see "EXLA is using GPU" or similar during compilation.

### macOS (Apple Silicon with Metal)

```bash
# Metal support is automatic on Apple Silicon
# XLA_TARGET is not needed (defaults to gpu)
# Just ensure you have Xcode installed

xcode-select --install

# Then compile normally
mix compile
```

### WSL2 Specific

If using WSL2 with GPU access:

```bash
# Inside WSL2, follow Linux/WSL2 setup above
# CUDA must be installed IN WSL2, not Windows
# Verify GPU is visible
nvidia-smi  # Should show GPU in WSL2
```

## Verify GPU Setup

After environment setup, confirm ExLA is using GPU:

```bash
# Check if ExLA recognizes GPU
mix run -e "
IO.inspect(EXLA.Client.list(), label: 'Available clients')
IO.inspect(EXLA.Client.memory_info(:default), label: 'GPU memory')
"

# Expected output:
# Available clients: [:cuda] or [:rocm]
# GPU memory: %{allocated: X, reserved: Y}

# If you see "cpu" instead, XLA_TARGET was not set correctly
# Re-run: export XLA_TARGET=cuda12 && mix compile --force
```

Verify with a quick tensor operation:

```bash
mix run -e "
t = Nx.iota({1000, 1000}, type: :f32)
result = Nx.sum(t)
IO.inspect(result, label: 'Sum result')
"

# Should complete quickly (~1-10ms for GPU)
# If slow (>500ms), running on CPU - fix XLA_TARGET
```

## Running GPU Tests

### Quick Sanity Check

```bash
# Verify ExLA CUDA is working
mix test test/exla_sanity_test.exs --exclude skip

# Expected: Tests pass with CUDA ops
```

### Bumblebee API Validation

```bash
# Test actual Bumblebee model loading and inference
mix test.gpu test/bumblebee_api_test.exs

# This will:
# - Download CLIP, VAE, UNet models from HuggingFace (~10-20GB)
# - Cache in /tmp/see-through-models
# - Run actual inference
# - Report shape/type information
```

### ModelServing Layer Testing

```bash
# Test the ModelServing abstraction with real models
mix test.gpu test/model_serving_test.exs

# Validates:
# - load_model/3 with real Bumblebee
# - load_tokenizer/2 with real tokenizers
# - run_inference/2 with actual models
# - encode_text/2 with actual tokenization
```

### Pipeline End-to-End

```bash
# Full pipeline E2E tests
mix test.gpu test/pipeline_e2e_test.exs

# Tests:
# - CLIP encoding (text → 1792-dim embeddings)
# - VAE encode (image → latent space)
# - VAE decode (latent → image)
# - Marigold depth (image → depth map)
# - UNet forward (latent → noise prediction)
# - Dual-pass LayerDiff (body + head passes)
```

### Full GPU Test Suite

```bash
# Run all GPU tests
mix test.gpu

# Expected output:
# - ModelCache tests: 5 passing
# - ModelServing tests: 3 passing (4 GPU)
# - Bumblebee API tests: 0-6 (depends on models)
# - Pipeline E2E tests: 0-8 (depends on GPU memory)
```

## Troubleshooting

### XLA_TARGET Not Set (Most Common)

```bash
# Error: Bumblebee models run on CPU despite having GPU
# Or: Tests skip GPU paths silently

# Solution: Set XLA_TARGET BEFORE mix compile
nvcc --version  # Check CUDA version
export XLA_TARGET=cuda12  # or cuda13

# IMPORTANT: Clean and recompile
mix deps.clean --all
mix compile

# Verify GPU is being used
mix run -e "IO.inspect(EXLA.Client.memory_info(:default))"
# Should show GPU memory info, not error
```

### CUDA Not Found

```bash
# Error: could not find CUDA libraries
# Or: CUDA_PATH not found

# Solution 1: Verify CUDA installation
nvidia-smi  # Check if GPU is visible
nvcc --version  # Check if compiler is available

# Solution 2: If CUDA is in non-standard location
export CUDA_PATH=/path/to/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export XLA_TARGET=cuda12

# Solution 3: Install CUDA if not found
# https://developer.nvidia.com/cuda-downloads
```

### XLA_TARGET Mismatch

```bash
# Error: CUDA version mismatch or API version incompatible
# Symptoms: Model inference fails with CUDA errors

# Solution: Verify CUDA version matches XLA_TARGET
nvcc --version  # Shows actual CUDA version
echo $XLA_TARGET  # Shows configured target

# If mismatch:
# - For CUDA 12.1-12.x: export XLA_TARGET=cuda12
# - For CUDA 13.0+: export XLA_TARGET=cuda13

# Then clean and rebuild
mix deps.clean xla --build
mix compile --force
```

### Out of Memory (OOM)

```bash
# Error: RuntimeError: CUDA out of memory
# Solution: Reduce batch size or model size

# Edit mix.exs to use smaller models:
config :see_through_burrito,
  clip_model: "openai/clip-vit-base-patch32",  # Smaller than vit-large
  vae_model: "stabilityai/sd-vae-ft-ema"        # Alternative VAE
```

### Model Download Timeout

```bash
# Error: timeout downloading from HuggingFace
# Solution: Increase timeout or pre-download

# Pre-download models:
mix run -e "
SeeThroughBurrito.ModelServing.load_model(\"openai/clip-vit-large-patch14\", :clip_text)
SeeThroughBurrito.ModelServing.load_model(\"stabilityai/sd-vae-ft-mse\", :vae)
"
```

### Bumblebee API Mismatch

```bash
# Error: Bumblebee.apply_model/2 undefined
# Solution: ModelServing will try fallback APIs

# Debug API mismatch:
mix run -e "
{:ok, model} = Bumblebee.load_model({:hf, \"test/model\"}, type: :clip_text)
IO.inspect(model.__struct__, label: \"Model type\")
IO.inspect(Map.keys(model), label: \"Model fields\")
"
```

## Performance Tuning

### Memory Usage

```bash
# Monitor CUDA memory
watch -n 0.1 nvidia-smi

# Reduce memory footprint:
# 1. Use half precision (fp16)
# 2. Reduce image resolution
# 3. Use smaller models
```

### Inference Speed

```bash
# Benchmark inference time
mix run benchmarks/inference.exs

# Expected performance (RTX 3090):
# - CLIP encoding: ~50-100ms per tag
# - VAE encode: ~200-300ms for 512x512 image
# - VAE decode: ~300-400ms from latent
# - UNet forward: ~100-200ms per step
# - Full dual-pass: ~30-60s (30 steps)
```

### Batch Processing

```bash
# For multiple images/prompts, batch to GPU
# Instead of:
Enum.map(tags, &encode_tag/1)

# Do this (groups 64 tags per batch):
tags
|> Enum.chunk_every(64)
|> Enum.map(&batch_encode/1)
```

## Next Steps After GPU Testing

1. **Validate ModelServing Output**
   - Confirm real inference shapes match expected
   - Document any API mismatches
   - Update fallback patterns if needed

2. **Integrate Diffusion Loop**
   - Wire Scheduler into UNet.pass/7
   - Test diffusion step iteration
   - Validate noise prediction

3. **Test Dual-Pass LayerDiff**
   - Run body pass (13 tags)
   - Run head pass (11 tags)
   - Verify layer outputs

4. **End-to-End Pipeline**
   - Load real anime image
   - Run full pipeline start-to-finish
   - Export SVG with layers
   - Compare vs see-through-cpp output

## Reference

- ExLA GPU Guide: https://hexdocs.pm/exla
- Bumblebee Docs: https://hexdocs.pm/bumblebee
- CUDA Setup: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- Model Zoo: https://huggingface.co/models?library=transformers
