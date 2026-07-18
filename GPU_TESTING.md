# GPU Testing Guide

## Prerequisites

1. **NVIDIA GPU** with CUDA compute capability 5.0+
   - Minimum: GTX 750 (2GB VRAM for basic inference)
   - Recommended: RTX 3060+ (12GB+ VRAM for real workloads)
   - Required: 24GB+ VRAM for dual-pass LayerDiff + Marigold

2. **CUDA Toolkit** 11.0+
   ```bash
   nvidia-smi  # Verify installation
   ```

3. **cuDNN** 8.0+
   ```bash
   # Part of CUDA installation
   ls /usr/local/cuda/lib64/libcudnn*
   ```

4. **Elixir & Mix**
   ```bash
   elixir --version  # Should be 1.15+
   mix --version
   ```

## Environment Setup

### Linux/WSL2

```bash
# Set CUDA paths
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Verify ExLA can find CUDA
mix compile
```

### macOS (Metal)

```bash
# Metal support is automatic on Apple Silicon
# No additional setup needed
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

### CUDA Not Found

```bash
# Error: could not find CUDA libraries
# Solution: Verify CUDA installation
nvidia-smi

# If not found, install CUDA
# https://developer.nvidia.com/cuda-downloads
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
