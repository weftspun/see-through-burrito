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
      {:nx, "~> 0.9"},
      {:exla, "~> 0.9"},

      # ML model management
      {:bumblebee, "~> 0.5"},

      # Executable packaging
      {:burrito, "~> 0.5"},

      # Property-based testing
      {:prop_check, "~> 1.4", only: :test},

      # HTTP client for model downloads
      {:httpoison, "~> 2.0"},
      {:jason, "~> 1.4"},

      # Image processing
      {:image, "~> 0.35"},

      # SVG output for layer export
      {:xmlbuilder, "~> 3.0"},
      {:vega_lite, "~> 0.1"}
    ]
  end
end
