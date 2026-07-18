# Critical Blockers for GPU Testing

## Status: Framework Complete, Inference Blocked on API Knowledge

All CPU tests passing (21/21). GPU tests ready but blocked on:

### 1. **Bumblebee 0.7.0 API Uncertainty**

Exact Bumblebee function signatures unknown. Need to validate via GPU testing:

#### CLIP Text Encoder
```elixir
# What we expect:
{:ok, embeddings} = Bumblebee.load_model({:hf, model_id}, type: :text, ...)
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_id}, ...)
{:ok, tokens} = tokenizer.encode(text)  # OR: Bumblebee.apply_tokenizer(tokenizer, text)
{:ok, embeddings} = model.predict(tokens)  # OR: Bumblebee.apply_model(model, tokens)

# Unknowns:
- Exact module paths and function names
- Return types (tuples vs direct values)
- Optional parameters
```

**Location**: `lib/model_serving.ex:106-109` (try_inference_apis/2)

#### VAE Encode/Decode
```elixir
# What we expect:
{:ok, vae} = Bumblebee.load_model({:hf, model_id}, type: :autoencoder, ...)
{:ok, latents} = vae.predict(image)  # or Bumblebee.apply_model
{:ok, image} = vae.predict(latents)

# Unknowns:
- Model loading signature
- Inference function names
- Input preprocessing requirements
```

**Location**: `lib/vae.ex:92-114` (run_vae_encoding/decoding)

#### LayerDiff UNet
```elixir
# What we expect:
{:ok, unet} = Bumblebee.load_model({:hf, model_id}, type: :unet, ...)
{:ok, noise} = unet.predict({latents, timestep, embeddings, page_rgb})

# Unknowns:
- Model loading for UNet variant
- Frame-conditioning input format
- Output shape and type
```

**Location**: `lib/unet.ex:55-66` (forward/6)

### 2. **Nx Random Number Generation**

Unknown API in this Nx 0.12.1 version:

```elixir
# Error in run_dual_pass_diffusion:
latents = Nx.random_normal({1, 64, 64, 4})  # UNDEFINED

# Alternatives to investigate:
- Nx.Random.normal(key, shape)  # if key-based
- Nx.random_normal(...)  # if direct
- Nx.Random.Generator pattern
- ExLA-specific random ops
```

**Location**: `lib/diffusion.ex:178`

### 3. **Scheduler Step Function Signature**

Current scheduler has:
```elixir
def dpm_solver_step(scheduler, sample, eps_pred, noise)  # 4 params
```

Diffusion loop calls:
```elixir
dpm_solver_step(sched_state, guided_noise, step)  # 3 params - MISMATCH
```

**Issue**: Don't know if noise should be included or what actual iteration looks like

**Location**: `lib/scheduler.ex:127` vs `lib/diffusion.ex:113` (commented out)

### 4. **Enum.find_map Availability**

```elixir
# Used in ModelServing but may not exist in Elixir 1.20
result = Enum.find_map(attempts, fn attempt -> attempt.() end)

# Alternatives:
- Enum.find + mapping pattern
- Custom recursion
- Stream-based approach
```

**Location**: `lib/model_serving.ex:109`

## Resolution Path

### Step 1: GPU Environment (Currently Blocked)
- Fix CUDA library paths OR rebuild XLA OR use conda
- Once GPU works: run `mix test.gpu test/bumblebee_api_test.exs`

### Step 2: API Discovery
Run individual GPU tests to observe:
```bash
# In GPU environment:
mix test.gpu test/bumblebee_api_test.exs

# Observe:
- What Bumblebee.load_model actually returns
- What tokenizer.encode returns
- What model inference returns
```

### Step 3: Wire Up Real Implementation
Update ModelServing.try_inference_apis to use discovered APIs:
- Replace placeholder returns with actual calls
- Update error handling based on real API behaviors
- Add model-specific preprocessing if needed

### Step 4: Complete Diffusion Loop
With real APIs known:
- Fix Scheduler.dpm_solver_step calls
- Implement random latent generation
- Wire Bumblebee.Diffusion if available

## Test Readiness

All test stubs created and marked `@skip`:
- `test/bumblebee_api_test.exs` - 6 tests for API discovery
- `test/model_serving_test.exs` - 4 GPU tests
- `test/pipeline_e2e_test.exs` - 8 end-to-end tests
- `test/diffusion_test.exs` - 2 tests (1 passing on CPU)

Tests will pass once API calls are wired correctly.

## Development Workflow

For someone with GPU access:

```bash
# 1. Fix GPU environment
bash setup_gpu.sh
source .env.gpu

# 2. Run API discovery tests
mix test.gpu test/bumblebee_api_test.exs

# 3. Take notes on actual API returns
# 4. Update ModelServing based on observations
# 5. Run full test suite
mix test.gpu

# 6. Wire diffusion loop
# 7. Run end-to-end tests
mix test.gpu test/pipeline_e2e_test.exs
```

## Current Status

| Component | Status | Blocker |
|-----------|--------|---------|
| CPU tests | ✅ 21/21 passing | None |
| GPU tests | ⏳ Ready to run | GPU environment + CUDA libs |
| CLIP module | 🟡 Stub complete | Bumblebee.load_model signature |
| VAE module | 🟡 Stub complete | Bumblebee inference API |
| UNet module | 🟡 Stub complete | Frame-conditioning format |
| Scheduler | ✅ Complete | Diffusion loop integration |
| Diffusion loop | 🟡 Skeleton complete | Scheduler step + Nx.random |
| ModelServing | 🟡 Multi-API fallback ready | Real API discovery |

**Next person to have GPU access should start with: GPU_TESTING.md + BLOCKERS.md + bumblebee_api_test.exs**
