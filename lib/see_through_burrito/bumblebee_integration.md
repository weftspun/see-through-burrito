# Bumblebee 0.7.0 Integration Guide

## Strategy

Replace placeholder `:not_implemented` stubs with real Bumblebee model loading and inference.

## Model Loading Pattern

### CLIP Text Encoder

```elixir
# Load CLIP-L (ViT-L)
{:ok, model_l} = Bumblebee.load_model(
  {:hf, "openai/clip-vit-large-patch14"},
  architecture: :clip_text_model,
  type: :text
)

# Load CLIP-G (ViT-G, optional, larger)
{:ok, model_g} = Bumblebee.load_model(
  {:hf, "openai/clip-vit-g-14"},
  architecture: :clip_text_model,
  type: :text
)

# Load tokenizers
{:ok, tokenizer_l} = Bumblebee.load_tokenizer(
  {:hf, "openai/clip-vit-large-patch14"},
  cache_dir: cache_dir
)
```

### VAE (Stable Diffusion)

```elixir
{:ok, vae} = Bumblebee.load_model(
  {:hf, "stabilityai/sd-vae-ft-mse"},
  architecture: :autoencoder,
  type: :vae
)
```

### LayerDiff UNet

```elixir
{:ok, unet} = Bumblebee.load_model(
  {:hf, "shitagaki-lab/layerdiff-unet"},
  architecture: :unet,
  type: :unet
)
```

### Marigold Depth

```elixir
{:ok, depth_model} = Bumblebee.load_model(
  {:hf, "prs-eth/marigold-v1"},
  type: :depth_estimation
)
```

## Inference Pattern

### Text Encoding (CLIP)

```elixir
# Tokenize text
{:ok, token_ids} = Bumblebee.Tokenizers.encode(tokenizer, "anime face")

# Run model (returns embeddings)
{:ok, embeddings} = Bumblebee.apply_model(model, {:text_encoder_input, token_ids})

# embeddings shape: {seq_len, embedding_dim}
# For CLIP: {77, 768} for ViT-L, {77, 1024} for ViT-G
```

### Image Encoding (VAE)

```elixir
# Prepare image (H, W, 3) normalized to [-1, 1]
{:ok, latents} = Bumblebee.apply_model(vae, {:encoder, image_tensor})

# latents shape: {H/8, W/8, 4}
```

### Diffusion (UNet)

```elixir
# Scale model input
scaled_latents = scheduler.scale_model_input(latents, sigma)

# Prepare context (embeddings, conditioning)
context = {embeddings, page_rgb, page_alpha}

# Run forward pass
{:ok, noise_pred} = Bumblebee.apply_model(unet, 
  {:unet_diffusion_input, scaled_latents, timestep, context}
)

# noise_pred shape: same as latents {batch, H/8, W/8, 4}
```

## Challenges & Solutions

### 1. Dynamic Model Loading

**Problem**: Models must be loaded once, not every request.

**Solution**: Use GenServer to cache loaded models.

```elixir
# supervisor child spec
{SeeThroughBurrito.ModelCache, [cache_dir: "/tmp/models"]}
```

### 2. Tokenizer Integration

**Problem**: Bumblebee.Tokenizers API may differ from expected.

**Solution**: Create adapter layer that handles API variations.

### 3. Batch Processing

**Problem**: Some models expect batched input, others single samples.

**Solution**: Implement tensor_ops tile_to_batch/1 and unbatch_first/1.

### 4. Device Placement (GPU/CPU)

**Problem**: ExLA defaults to CPU; must force CUDA/Metal.

**Solution**: Configure in config/config.exs:

```elixir
config :exla, 
  clients: [cuda: [platform: :cuda]], 
  default_client: :cuda
```

### 5. Scheduler Integration

**Problem**: Bumblebee schedulers may not match DPM-Solver++.

**Solution**: Keep custom scheduler.ex (already ported from C++).

## Implementation Checklist

- [ ] CLIP text encoder
  - [ ] load_model/2 → Bumblebee.load_model/3
  - [ ] load_tokenizer/2 → Bumblebee.Tokenizers.tokenize/2
  - [ ] run_inference/2 → Bumblebee.apply_model/2

- [ ] VAE encode/decode
  - [ ] load_vae_model/2 → Bumblebee.load_model/3
  - [ ] run_vae_encoding/2 → Bumblebee.apply_model/2
  - [ ] run_vae_decoding/2 → Bumblebee.apply_model/2

- [ ] LayerDiff UNet
  - [ ] load_model/1 → Bumblebee.load_model/3
  - [ ] forward/6 → Bumblebee.apply_model/2

- [ ] Marigold Depth
  - [ ] load_model/2 → Bumblebee.load_model/3
  - [ ] run_depth_inference/3 → Bumblebee.apply_model/2 + DDIM scheduler

- [ ] Model cache (GenServer)

## References

- Bumblebee docs: https://github.com/elixir-nx/bumblebee
- Model hub: https://huggingface.co
- ExLA GPU setup: https://github.com/elixir-nx/elixir_nx
