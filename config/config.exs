import Config

# ExLA configuration - GPU-first (CUDA/Metal), no CPU fallback
# Supported platforms:
# - CUDA (NVIDIA GPUs on Linux/Windows)
# - Metal-EXLA (Apple Silicon on macOS)
config :nx,
  default_backend: EXLA.Backend,
  default_defn_options: [compiler: EXLA]

# GPU-only clients. No CPU fallback.
config :exla,
  clients: [
    # Linux/Windows: CUDA
    cuda: [platform: :cuda],
    # macOS: Metal
    metal: [platform: :metal]
  ],
  default_client: :cuda

# HTTP timeouts for model downloads
config :httpoison, timeout: 300_000

import_config("#{Mix.env()}.exs")
