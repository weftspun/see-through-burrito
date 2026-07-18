# START_HERE.md - For Next Developer with GPU Access

This is your entry point. Follow these steps in order.

## Prerequisites (Before You Start)

- ✅ NVIDIA GPU with CUDA Compute Capability 3.5+
- ✅ CUDA toolkit 12.x or 13.x installed
- ✅ Elixir 1.20.2+ installed
- ✅ This repository cloned

**No GPU? Skip to "CPU Development" section at bottom.**

---

## Phase 1: GPU Setup (15 minutes)

### Step 1.1: Verify GPU

```bash
nvcc --version
# Expected: Cuda compilation tools, release 13.x or 12.x

nvidia-smi
# Expected: Shows GPU name, memory, driver version
```

If either fails, follow [GPU_SETUP_REQUIRED.md](GPU_SETUP_REQUIRED.md) first.

### Step 1.2: Run Automated Setup

```bash
bash setup_gpu.sh
```

This script will:
- Detect CUDA version
- Auto-configure XLA_TARGET
- Verify GPU connectivity
- Compile project with GPU support
- Run verification tests

**Output should end with: ✅ GPU Setup Complete!**

### Step 1.3: Verify Setup

```bash
source .env.gpu
export XLA_TARGET=cuda13  # or cuda12

mix run -e "IO.inspect(EXLA.Client.list())"
# Expected: [:cuda] or [:metal]
```

---

## Phase 2: API Discovery Tests (30 minutes)

These tests will reveal exact Bumblebee 0.7.0 API signatures.

### Step 2.1: Run Discovery Tests

```bash
mix test test/bumblebee_api_test.exs --no-exclude-tags
```

**Important:** Examine the test output carefully. Look for:
- What `Bumblebee.load_model()` actually returns
- What tokenizer methods are available
- What model inference methods return
- Any errors or unexpected API shapes

### Step 2.2: Document Findings

Create a file `API_FINDINGS.md` with observations:

```markdown
## Bumblebee 0.7.0 API Findings

### Model Loading
- Function: Bumblebee.load_model/2 or /3
- Return: {:ok, model} or model directly
- Signature example:
  {:ok, model} = Bumblebee.load_model({:hf, "model-id"})

### Inference
- Function: model.predict/1 or Bumblebee.apply_model/2
- Input: tensor or batch
- Output: tensor or {:ok, tensor}

### Issues Found
- (List any API surprises or errors here)
```

### Step 2.3: Update ModelServing

Based on findings, update `lib/model_serving.ex` lines 106-150:

```elixir
# OLD (placeholder):
{:error, :not_implemented}

# NEW (based on actual API):
case Bumblebee.load_model({:hf, model_id}) do
  {:ok, model} -> {:ok, model}
  model when is_struct(model) -> {:ok, model}
  error -> error
end
```

---

## Phase 3: Run Full Test Suite (45 minutes)

### Step 3.1: Basic Tests (Should Already Pass)

```bash
mix test --exclude skip
# Expected: 21 passed, 51 excluded
```

### Step 3.2: GPU Tests (Now Should Work)

```bash
mix test
# Expected: 72 passed total (21 CPU + 51 GPU)
```

If tests fail, check:
- GPU memory (72 total tests may need 12-16GB)
- Which specific test fails
- Error message in BLOCKERS.md may have resolution

### Step 3.3: Pipeline End-to-End Test

```bash
mix test test/pipeline_e2e_test.exs
# Expected: 8 tests passing
# Shows full decomposition pipeline works
```

---

## Phase 4: Wire Diffusion Loop (2 hours)

The main blocker is the diffusion loop integration.

### Step 4.1: Understand Current State

```bash
cat lib/diffusion.ex | grep -A 30 "diffusion_loop"
```

Current: Returns input latents unchanged (placeholder)
Needed: Actual diffusion iteration with scheduler

### Step 4.2: Integrate Scheduler

Update `lib/diffusion.ex:97-146`:

```elixir
# Current:
def diffusion_loop(scheduler, latents, embeddings, guidance, steps, page_info) do
  {:ok, latents}  # Placeholder
end

# Needed:
def diffusion_loop(scheduler, latents, embeddings, guidance, steps, page_info) do
  Enum.reduce(0..(steps - 1), {:ok, latents}, fn step, {:ok, lat} ->
    # 1. Get noise prediction from UNet
    {:ok, noise} = Unet.forward(unet_model, lat, scheduler.timesteps[step], embeddings, page_info)
    
    # 2. Apply guidance
    guided_noise = apply_classifier_free_guidance(noise, guidance)
    
    # 3. Step scheduler
    {:ok, next_lat, next_scheduler} = Scheduler.dpm_solver_step(scheduler, lat, guided_noise, noise)
    
    {:ok, next_lat}
  end)
end
```

### Step 4.3: Test Integration

```bash
mix test test/diffusion_test.exs
# Expected: 2 tests passing
```

---

## Phase 5: Test with Real Image (30 minutes)

### Step 5.1: Prepare Input

```bash
# Get a sample anime image (720p or larger)
wget -O sample.png https://example.com/anime.png

# Or use your own image
cp my_image.png sample.png
```

### Step 5.2: Build Executable

```bash
mix escript.build
```

### Step 5.3: Run Full Pipeline

```bash
time ./see_through_burrito \
  --input sample.png \
  --output layers.svg \
  --steps 30 \
  --guidance 7.5 \
  --width 1024 \
  --height 1024
```

Expected output:
- `layers.svg` — SVG file with all 24 layers
- Execution time: 15-30 seconds (RTX 4090)
- Console logs showing progress

### Step 5.4: Verify Output

```bash
# Check file exists
ls -lh layers.svg

# View in browser
firefox layers.svg

# Verify structure
file layers.svg
# Expected: SVG (Scalable Vector Graphics) image

# Check embedded images
unzip -l layers.svg 2>/dev/null || echo "Not a ZIP (correct for SVG)"
```

---

## Phase 6: Performance Benchmark (20 minutes)

### Step 6.1: Measure Baseline

```bash
time ./see_through_burrito -i sample.png -o test1.svg --steps 30
time ./see_through_burrito -i sample.png -o test2.svg --steps 30
time ./see_through_burrito -i sample.png -o test3.svg --steps 30

# Average the three runs
```

### Step 6.2: Compare vs Reference

Expected vs see-through-cpp:
- Similar inference time (within 2x)
- Identical layer decomposition quality
- Same 24 semantic layers

### Step 6.3: Document Results

Update `PERFORMANCE.md`:

```markdown
## Benchmark Results

Hardware: [Your GPU Model]
Date: 2026-07-XX

### Pipeline Performance
- Preprocessing: XXms
- CLIP encoding: XXms
- VAE encode: XXms
- Layer decomposition: XXms
- VAE decode: XXms
- Depth estimation: XXms
- Post-processing: XXms
- SVG export: XXms

Total: XXs per image

### Memory Usage
- Peak GPU memory: XXGB
- Peak CPU memory: XXMB
```

---

## Phase 7: Create GPU Test Report (20 minutes)

### Step 7.1: Document Your Environment

Create `GPU_TEST_REPORT.md`:

```markdown
# GPU Test Report - [Date]

## Environment
- CUDA Version: X.X.X
- GPU Model: [Model Name]
- GPU Memory: XXG
- Driver Version: XXX.XX
- Elixir Version: 1.20.2
- Erlang Version: XX.X

## Test Results

### Compilation
- ✅ Setup GPU: PASS
- ✅ Dependencies: PASS
- ✅ Compilation: PASS (XXs)

### Unit Tests
- ✅ CPU tests: 21 passed
- ✅ GPU tests: 51 passed
- ✅ Total: 72 passed

### Integration Tests
- ✅ Pipeline E2E: 8 passed
- ✅ Model loading: PASS
- ✅ Inference: PASS

### Real-World Test
- ✅ Image decomposition: PASS
- ✅ SVG export: PASS
- ✅ 24 layers extracted: PASS
- ✅ Performance: XXs/image

## Issues Encountered
- (List any issues and resolutions)

## Recommendations
- (Any suggestions for next developer)
```

### Step 7.2: Commit Your Work

```bash
git add GPU_TEST_REPORT.md API_FINDINGS.md PERFORMANCE.md
git commit -m "GPU validation complete on [GPU model]

All 72 tests passing. Diffusion loop integrated and working.
Real image decomposition verified. Performance within expectations.

See GPU_TEST_REPORT.md for detailed results."
```

---

## Troubleshooting

### Tests Fail with OOM (Out of Memory)

Reduce batch size or step count:
```bash
mix test test/pipeline_e2e_test.exs --exclude skip
# If fails, reduce in test/pipeline_e2e_test.exs:
# steps: 10  (was 30)
# width: 512  (was 1024)
```

### Diffusion Loop Doesn't Converge

Check:
1. Scheduler initialization (dpm_solver_init)
2. UNet model output shape
3. Guidance scale (7.5-15.0 typical)
4. Number of steps (30-50 typical)

### SVG Output is Blank

Check:
1. Layer masks are not empty (threshold too high?)
2. Base64 encoding of images
3. SVG viewBox dimensions

### Performance Much Slower Than Expected

Verify:
```bash
mix run -e "IO.inspect(Nx.default_backend())"
# Should be: EXLA.Backend

mix run -e "IO.inspect(EXLA.Client.list())"
# Should show: [:cuda]

# If CPU backend, recompile:
mix clean
export XLA_TARGET=cuda13
mix compile --force
```

---

## Success Criteria

You're done when:

- ✅ GPU setup complete and verified
- ✅ All 72 tests passing
- ✅ Real image decomposition works
- ✅ Performance within 2x of C++ version
- ✅ SVG output has all 24 layers
- ✅ Report committed to git
- ✅ No Dialyzer defects (mix dialyzer)

---

## CPU Development (No GPU)

If you don't have GPU access, you can still develop:

```bash
# Reconfigure for CPU testing
export SKIP_GPU=1
mix test --exclude skip

# This runs 21 CPU-compatible tests
# GPU tests marked @skip will be excluded
```

Limitations:
- Can't test actual Bumblebee models
- Can't validate diffusion pipeline
- Can test architecture and logic only

---

## Next Steps After Success

1. **Performance Tuning**
   - Profile with `:fprof` or `:eprof`
   - Identify bottlenecks
   - Optimize hot paths

2. **Feature Enhancements**
   - Batch processing
   - Frame conditioning
   - Custom layer definitions

3. **Deployment**
   - Build with Burrito: `mix burrito.build`
   - Test on target system
   - Document deployment procedure

4. **Integration**
   - Create PR to weftspun/see-through-elixir
   - Merge to main
   - Tag release

---

## Quick Reference

```bash
# Full setup
bash setup_gpu.sh

# Tests
mix test                          # All tests
mix test --exclude skip           # CPU only
mix test test/bumblebee_api_test.exs  # API discovery
mix test test/pipeline_e2e_test.exs   # Full pipeline

# Compile
mix compile --force

# Run
mix escript.build
./see_through_burrito -i input.png -o output.svg

# Check
mix dialyzer
mix format --check-formatted
mix credo
```

---

## Questions?

- See `BLOCKERS.md` for known issues
- See `GPU_TESTING.md` for detailed GPU testing guide
- See `READINESS_REPORT.md` for overall project status
- Check `test/*.exs` for example usage

Good luck! 🚀
