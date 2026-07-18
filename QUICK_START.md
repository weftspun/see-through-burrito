# Quick Start: see-through-burrito Development

Welcome! This guide gets you from zero to running in ~10 minutes.

## Prerequisites

- **Elixir 1.16+** and **Erlang 26+**
  ```bash
  # Check versions
  elixir --version
  # Should be >= 1.16.0
  ```

- **CUDA 12.x or 13.x** (GPU-first design)
  ```bash
  nvcc --version
  # Should show CUDA 12.x or 13.x
  ```

## 1. Clone & Setup (2 min)

```bash
git clone https://github.com/weftspun/see-through-burrito
cd see-through-burrito
mix deps.get
```

## 2. Configure GPU (2 min)

**Set XLA target BEFORE compiling:**

```bash
# Detect CUDA version
nvcc --version  # Note: 12 or 13

# Set XLA_TARGET based on CUDA version
export XLA_TARGET=cuda13  # For CUDA 13.x
# OR
export XLA_TARGET=cuda12  # For CUDA 12.x

# Compile with GPU support
mix compile
```

**⚠️ CRITICAL**: XLA_TARGET must be set BEFORE `mix compile` or compilation will fail.

## 3. Verify GPU Setup (1 min)

```bash
# Test GPU connectivity
mix run -e "IO.inspect(EXLA.Client.list())"
# Should output: [:cuda]

# Run sanity test
mix test test/exla_sanity_test.exs
# Should pass with GPU operations
```

## 4. Run CPU Tests (1 min)

All CPU-compatible operations pass without GPU:

```bash
mix test --exclude skip
# Result: 21 passed, 51 excluded
```

## 5. Run GPU Tests (if GPU available)

Discover Bumblebee APIs by running actual tests:

```bash
mix test.gpu test/bumblebee_api_test.exs
# Observes actual return types from Bumblebee 0.7.0
# Output becomes input for ModelServing.try_inference_apis/2
```

## Project Structure

```
lib/
├── see_through_burrito.ex          # Main orchestration
├── adapters.ex                     # Behavior contracts (5 traits)
├── model_serving.ex                # Unified Bumblebee API layer
│
├── clip.ex                         # Text encoding
├── vae.ex                          # Image compression
├── unet.ex                         # Layer decomposition
├── marigold.ex                     # Depth estimation
│
├── pipeline.ex                     # Workflow orchestration
├── diffusion.ex                    # Diffusion scheduler loop
├── scheduler.ex                    # DPM-Solver++, DDIM
├── postproc.ex                     # Post-processing
│
├── model_cache.ex                  # Model caching GenServer
├── tensor_ops.ex                   # GPU tensor operations
├── images.ex                       # Image I/O
└── svg_export.ex                   # SVG output

test/
├── bumblebee_api_test.exs         # API discovery (6 @skip)
├── model_serving_test.exs         # ModelServing layer (8)
├── pipeline_e2e_test.exs          # E2E pipeline (8 @skip)
├── diffusion_test.exs             # Diffusion (2)
└── 9 more test files              # Full coverage

docs/
├── GPU_TESTING.md                 # 5-min GPU setup guide
├── BLOCKERS.md                    # Known API unknowns
├── READINESS_REPORT.md            # Project status (75%)
├── PORTING.md                     # Phase-by-phase progress
└── SESSION_SUMMARY.md             # Architecture decisions
```

## Key Concepts

### Hexagonal Architecture
All models (CLIP, VAE, UNet, Marigold) implement **Adapter** behavior contract. Each has two functions:

1. **load_model/2** — Model initialization via ModelServing
2. **run_inference/2** — Inference with multi-API fallback patterns

This enables:
- ✅ Independent unit testing via Mox
- ✅ Easy model substitution
- ✅ Graceful API version tolerance

### ModelServing Layer
Central abstraction for Bumblebee 0.7.0 APIs:

```elixir
# Load any model with consistent interface
SeeThroughBurrito.ModelServing.load_model(model_id, :clip_text, cache_dir: "...")
# Returns: {:ok, model} or {:error, reason}

# Run inference with multi-API fallback
SeeThroughBurrito.ModelServing.run_inference(model, input)
# Tries: Bumblebee.apply_model → Axon.predict → model.predict → ...
```

**See**: `lib/model_serving.ex:106-109` for fallback attempts

### Dependency Injection
All modules accept optional adapters via keyword opts:

```elixir
# Production (uses real Bumblebee)
SeeThroughBurrito.Diffusion.run_diffusion(image, prompt, [])

# Testing (uses mock)
SeeThroughBurrito.Diffusion.run_diffusion(image, prompt, 
  model_adapter: MockModelServing)
```

## Common Tasks

### Run All CPU Tests
```bash
mix test --exclude skip
```

### Run Specific Test File
```bash
mix test test/model_serving_test.exs
```

### Check Type Safety
```bash
mix dialyzer
# Should report: 0 defects
```

### Recompile with GPU
```bash
export XLA_TARGET=cuda13
rm -rf deps/_build
mix compile
```

### Clean CUDA Cache
```bash
rm -rf ~/.cache/xla
mix compile
```

## Troubleshooting

### "XLA_TARGET not set"
**Error**: `undefined reference to 'xla_client_*'`

**Fix**:
```bash
export XLA_TARGET=cuda13  # Before mix compile!
mix compile
```

### "NCCL not found"
**Error**: `libnccl.so.2: cannot open shared object`

**Fix** (choose one):
```bash
# Option 1: Install system NCCL + cuDNN
sudo apt-get install libnccl2 libcudnn9

# Option 2: Build XLA from source
export XLA_TARGET=cuda13
export XLA_BUILD=true
mix compile  # Takes 30-60 min

# Option 3: Use conda environment
bash setup_gpu.sh  # Automated
```

### "EXLA.Client.list() shows empty"
**Issue**: GPU not accessible to ExLA

**Debug**:
```bash
nvidia-smi          # Verify GPU visible
mix clean            # Clear compiled artifacts
mix compile          # Recompile with GPU
mix run -e "IO.inspect(EXLA.Client.list())"
```

### "Tests pass but inference returns :not_implemented"
**Expected**: All real inference is placeholder until GPU testing discovers Bumblebee APIs.

**Next step**: Run `mix test.gpu test/bumblebee_api_test.exs` to observe actual APIs, then update `ModelServing.try_inference_apis/2`.

## Architecture Files

If you need to understand the design:

- **[Hexagonal Architecture Decision](../.claude/CLAUDE.md)** — Why Ports & Adapters, why dependency injection
- **[GPU Testing Strategy](GPU_TESTING.md)** — How to validate GPU setup and run tests
- **[Known Blockers](BLOCKERS.md)** — Exact API unknowns blocking real inference
- **[Porting Progress](PORTING.md)** — Phase-by-phase completion status

## Next Steps

If you have **GPU access**:

1. **Verify GPU** (1 min)
   ```bash
   bash setup_gpu.sh
   source .env.gpu
   ```

2. **Discover Bumblebee APIs** (30 min)
   ```bash
   mix test.gpu test/bumblebee_api_test.exs
   # Note actual return types and function names
   ```

3. **Wire real implementations** (2 hours)
   - Update `ModelServing.try_inference_apis/2`
   - Update CLIP, VAE, UNet, Marigold load/inference
   - Update Diffusion scheduler integration

4. **Validate pipeline** (1 hour)
   ```bash
   mix test.gpu  # All 72 tests should pass
   ./see_through_burrito -i sample.png -o output.svg
   ```

If you **don't have GPU access**:

- ✅ All CPU tests pass (21/21)
- ✅ All architecture complete
- ✅ Full test suite staged (51 GPU tests marked @skip)
- ⏳ GPU testing is next developer's task

## Git Workflow

```bash
# Check status before committing
git status

# Create a feature branch
git checkout -b feature/your-feature

# Make changes and test
mix test
mix dialyzer

# Commit with descriptive message
git commit -m "Brief description

Detailed explanation of what changed and why."

# Push and create PR
git push origin feature/your-feature
```

## Getting Help

- **Code Questions**: Check `lib/*/` module docstrings (@doc attributes)
- **API Issues**: See `BLOCKERS.md` for known unknowns
- **Test Patterns**: Look at existing tests in `test/` directory
- **Architecture**: Read `.claude/CLAUDE.md` for design decisions

## Performance Notes

- **First run**: HuggingFace models auto-download (~2-5 GB)
- **Model caching**: Cached in `~/.cache/see-through-models` by default
- **Batch size**: Currently 1 image per run (parallelization pending)
- **Memory**: Requires ~24 GB VRAM for full pipeline with 1024×1024 images

---

**Status**: 75% complete, all architecture ready, 21/21 CPU tests passing, 0 Dialyzer defects

**Latest update**: Loop 7 — Documentation and API guide
