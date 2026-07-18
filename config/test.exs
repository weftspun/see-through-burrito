import Config

# For testing without GPU: use Host backend (EXLA compiles to CPU)
# On systems with CUDA toolkit, remove this to use GPU
config :nx,
  default_backend: Nx.BinaryBackend

config :exla,
  clients: [host: [platform: :host]],
  default_client: :host
