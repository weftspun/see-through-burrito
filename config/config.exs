import Config

# ExLA configuration - GPU first (CUDA), no CPU fallback
config :nx,
  default_backend: EXLA.Backend,
  default_defn_options: [compiler: EXLA]

config :exla,
  clients: [
    cuda: [platform: :cuda]
  ],
  default_client: :cuda

# Logger configuration removed - Elixir 1.20+ uses modern logger

# HTTP timeouts for model downloads
config :httpoison, timeout: 300_000

import_config("#{Mix.env()}.exs")
