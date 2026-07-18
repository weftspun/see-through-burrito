# see-through-burrito

Elixir implementation of [See-Through](https://github.com/weftspun/see-through) (Shitagaki Lab, SIGGRAPH 2026): decompose anime illustrations into up to 24 semantic layers using neural networks, with GPU acceleration via ExLA and self-contained executable packaging via Burrito.

## Architecture

- **ExLA + CUDA**: GPU-accelerated numerical computation (NVIDIA CUDA, no CPU fallback)
- **Bumblebee**: Hugging Face model loading and serving for diffusion models, VAE, text encoders
- **Layers**: Semantic decomposition using LayerDiff UNet into 24+ layer types
- **Depth**: Monocular depth estimation using Marigold
- **Inpainting**: Hole filling via LaMa or similar
- **Output**: SVG layers with embedded base64 images and metadata
- **Testing**: Property-based tests with PropCheck for numerical stability
- **Packaging**: Burrito for self-contained executable binaries

## Usage

### Compile and Build

```bash
mix deps.get
mix compile
```

### Run Locally (Development)

```bash
mix escript.build
./see_through_burrito --input anime.png --output output.svg
```

### Build Executable with Burrito

```bash
mix burrito.build
# Output: burrito-output/see_through_burrito_linux (or _macos, etc.)
```

### CLI Options

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

### Example

```bash
./see_through_burrito \
  --input anime.png \
  --output layers.svg \
  --steps 50 \
  --guidance 10.0
```

## Layer Output

The tool outputs an SVG file containing:
- **24 semantic layers**: background, face parts, hair, clothing, accessories, etc.
- **Base64 embedded images**: Each layer as an embedded PNG image
- **Depth map**: Monocular depth estimation as a pattern
- **Metadata**: Layer names, order, and export information
- **Manifest JSON**: `output.svg.json` with layer metadata

SVG is suitable for:
- Web display and interaction
- Layer manipulation in CSS/JS
- Vector graphics tools (Illustrator, Inkscape)
- Compositing and post-processing

## Development

### Run Tests

```bash
mix test
```

### Property-Based Tests

```bash
mix test test/prop_check_tests.exs
```

Tests verify:
- Tensor shape preservation
- Value range bounds
- Numerical stability
- Operation commutativity

### Project Structure

```
lib/
├── see_through_burrito.ex          # Main entry point
├── see_through_burrito/
│   ├── application.ex              # OTP app
│   ├── models.ex                   # Bumblebee model loading
│   ├── images.ex                   # Image I/O and preprocessing
│   ├── pipeline.ex                 # Main orchestration
│   ├── encoder.ex                  # VAE encoding/decoding
│   ├── layers.ex                   # Layer decomposition (LayerDiff)
│   ├── depth.ex                    # Depth estimation (Marigold)
│   ├── inpaint.ex                  # Inpainting (LaMa)
│   ├── svg_export.ex               # SVG output generation
│   └── cli.ex                      # CLI interface
test/
├── prop_check_tests.exs            # Property-based tests
└── see_through_burrito_test.exs    # Unit tests
config/
├── config.exs                      # ExLA GPU configuration
├── test.exs                        # Test-specific config
└── prod.exs                        # Production config
```

## GPU Configuration

ExLA is configured for CUDA by default (no CPU fallback):

```elixir
# config/config.exs
config :exla,
  clients: [cuda: [platform: :cuda]],
  default_client: :cuda
```

To use a different GPU backend (ROCm):
```elixir
config :exla,
  clients: [rocm: [platform: :rocm]],
  default_client: :rocm
```

## Model Weights

Models are downloaded automatically from Hugging Face on first run:
- `shitagaki-lab/layerdiff-unet` (LayerDiff diffusion model)
- `stabilityai/sd-vae-ft-mse` (VAE encoder/decoder)
- `prs-eth/marigold-v1` (Depth estimation)
- `facebook/lama` (Inpainting)
- CLIP text encoders (for prompting)

Cached in `$XDG_CACHE_HOME/huggingface` or `/tmp/see-through-models`.

## TODO

- [ ] Thorvg Rustler NIF for advanced SVG/vector rendering
- [ ] Optimize layer decomposition with frame-conditioned diffusion
- [ ] Implement parallel layer processing
- [ ] Add layer export to individual SVG files
- [ ] PSD layer output (optional)
- [ ] Web UI with warp output
- [ ] Benchmark GPU memory usage

## References

- [See-Through](https://github.com/weftspun/see-through) — Original PyTorch repo
- [see-through-cpp](https://github.com/weftspun/see-through-cpp) — C++/GGML port
- [Bumblebee](https://github.com/elixir-nx/bumblebee) — Elixir ML model serving
- [ExLA](https://github.com/elixir-nx/elixir_nx#exla) — GPU backend for Nx
- [Burrito](https://github.com/burrito-elixir/burrito) — Executable packaging

