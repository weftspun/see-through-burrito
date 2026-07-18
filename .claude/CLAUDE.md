# see-through-burrito Architecture

## Project Overview

Elixir implementation of See-Through: decompose anime illustrations into 24+ semantic layers using GPU-accelerated neural networks, with output as SVG.

**Key constraint**: GPU-first (CUDA/ROCm), no CPU fallback.

## Stack

- **ExLA**: GPU compute backend (CUDA by default, configured in config/config.exs)
- **Bumblebee**: Model serving (diffusion, VAE, text encoders via HuggingFace)
- **Burrito**: Self-contained executable packaging
- **Nx**: Numerical computing / tensor ops
- **PropCheck**: Property-based testing for numerical stability
- **XmlBuilder + JSON**: SVG + metadata export

## Module Organization

### Core Pipeline
- `pipeline.ex` — Main orchestration: preprocess → encode → diffuse → decode
- `encoder.ex` — VAE latent encode/decode
- `models.ex` — Bumblebee wrappers for model loading & inference

### Decomposition & Estimation
- `layers.ex` — Layer extraction using LayerDiff UNet (24 layers: background, face, hair, clothing, accessories, etc.)
- `depth.ex` — Monocular depth via Marigold (normalize, invert, compute normals)
- `inpaint.ex` — Hole filling via LaMa (morphological ops, blending)

### I/O
- `images.ex` — Image loading (Image.ex), tensor conversion, padding
- `svg_export.ex` — SVG generation with base64 embedded images, layer metadata, depth pattern
- `cli.ex` — escript CLI entry point

### Testing
- `test/prop_check_tests.exs` — Property-based tests: shape invariance, range bounds, numerical stability, monotonicity

## GPU Configuration

ExLA CUDA by default:
```elixir
# config/config.exs
config :exla, clients: [cuda: [platform: :cuda]], default_client: :cuda
```

No CPU fallback on purpose. To change:
- ROCm: `platform: :rocm`

## Layer List

24 semantic layers (exhaustive, from `layers.ex`):
- background, character_body, face, eyes, eyebrows, nose, mouth
- hair_front, hair_back, hair_sides, hair_bangs, hair_accessories
- neck, skin
- clothes_upper, clothes_lower, clothes_outerwear
- gloves, shoes, socks
- accessories_head, accessories_neck, accessories_hand, accessories_waist

## SVG Output

- One SVG per run: `output.svg`
- Embedded base64 PNG images (each layer)
- Metadata: layer order, names, depth pattern
- Manifest JSON: `output.svg.json` with layer metadata

## Property-Based Testing Strategy

Tests in `test/prop_check_tests.exs` verify:
1. **Shape preservation**: Ops don't alter tensor dimensions
2. **Range bounds**: Normalized outputs stay in [0,1]; finite checks
3. **Idempotence**: Double operations converge (opening→opening stable)
4. **Monotonicity**: Guidance scaling increases magnitude linearly
5. **Invertibility**: Double inversion returns to original (within tolerance)

Generators: matrix_data (random tensors), matrix_data_normalized (0-1), matrix_data_binary (0/1 masks).

## Known Limitations & TODOs

- **Thorvg Rustler NIF**: Not yet implemented; for advanced vector rendering in SVG
- **Parallel layer extraction**: Current implementation is sequential
- **Frame conditioning**: LayerDiff has cross-frame dependencies not yet leveraged
- **Model caching**: Currently /tmp/see-through-models; should use XDG_CACHE_HOME

## Build & Deploy

### Development
```bash
mix escript.build
./see_through_burrito -i input.png -o output.svg
```

### Production (Burrito)
```bash
mix burrito.build
# Creates burrito-output/see_through_burrito_linux, etc.
```

### CLI Flags
- `-i, --input` (required): Image path
- `-o, --output`: SVG output path (default: ./output.svg)
- `--steps`: Diffusion iterations (default: 30)
- `--guidance`: Guidance scale (default: 7.5)
- `--depth-steps`: Depth estimation steps (default: 4)
- `--width, --height`: Target resolution (default: 1024×1024)
- `--seed`: Random seed
- `--model-cache`: Cache directory for downloaded models

## Model Weights

Auto-downloaded on first run (HuggingFace):
- `shitagaki-lab/layerdiff-unet`
- `stabilityai/sd-vae-ft-mse`
- `prs-eth/marigold-v1`
- `facebook/lama`
- CLIP text encoders

Cached in `/tmp/see-through-models` (or via `--model-cache`).

## References

- **See-Through (PyTorch)**: https://github.com/weftspun/see-through
- **see-through-cpp**: https://github.com/weftspun/see-through-cpp
- **Bumblebee**: https://github.com/elixir-nx/bumblebee
- **ExLA**: https://github.com/elixir-nx/elixir_nx
- **Burrito**: https://github.com/burrito-elixir/burrito
