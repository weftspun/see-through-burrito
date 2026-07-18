defmodule SeeThroughBurrito.MixProject do
  use Mix.Project

  def project do
    [
      app: :see_through_burrito,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      escript: [main_module: SeeThroughBurrito.CLI]
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {SeeThroughBurrito.Application, []}
    ]
  end

  defp deps do
    [
      # Numerical computation (GPU-first, CUDA via ExLA)
      {:nx, "~> 0.12"},
      {:exla, "~> 0.12"},

      # ML model management (latest with Axon/Safetensors support)
      {:bumblebee, "~> 0.7.0"},
      {:axon, "~> 0.8.1"},
      {:safetensors, "~> 0.1.3"},
      {:tokenizers, "~> 0.5.1"},

      # Property-based testing
      {:stream_data, "~> 1.0", only: :test},

      # HTTP client for model downloads
      {:httpoison, "~> 2.0"},
      {:jason, "~> 1.4"},

      # Image processing
      {:image, "~> 0.35"},

      # SVG output for layer export
      {:xml_builder, "~> 2.3"}
    ]
  end
end
