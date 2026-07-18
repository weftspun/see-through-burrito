import Config

# ExLA configuration - GPU-MANDATORY (CUDA/Metal)
# No CPU fallback. Compilation will fail if XLA_TARGET not set.

unless System.get_env("XLA_TARGET") do
  raise """
  ❌ GPU REQUIRED: XLA_TARGET not set!

  Set GPU target before compiling:
    export XLA_TARGET=cuda13    # NVIDIA CUDA 13.x
    export XLA_TARGET=cuda12    # NVIDIA CUDA 12.x
    export XLA_TARGET=metal     # Apple Silicon (macOS)

  Then run:
    mix compile

  For WSL2 CUDA setup:
    bash setup_gpu.sh
  """
end

config :nx,
  default_backend: EXLA.Backend,
  default_defn_options: [compiler: EXLA]

# GPU-only clients. No CPU fallback.
config :exla,
  clients: [
    cuda: [platform: :cuda],
    metal: [platform: :metal]
  ],
  default_client: :cuda

# HTTP timeouts for model downloads
config :httpoison, timeout: 300_000

import_config("#{Mix.env()}.exs")
