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
