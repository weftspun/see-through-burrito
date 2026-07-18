defmodule SeeThroughBurrito.MixProject do
  use Mix.Project

  def project do
    [
      app: :see_through_burrito,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      escript: [main_module: SeeThroughBurrito.CLI],
      aliases: aliases(),
      dialyzer: [
        plt_add_apps: [:ex_unit],
        plt_core_path: "priv/plts",
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"}
      ]
    ]
  end

  defp aliases do
    [
      test: "test --exclude skip",
      "test.gpu": "test --include skip",
      bench: "run benchmarks/tensor_ops.exs",
      release: "burrito.build",
      cli: "escript.build",
      dialyzer: "dialyzer --format short"
    ]
  end

  def cli do
    [preferred_envs: ["test.gpu": :test, bench: :dev, release: :prod, cli: :dev]]
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

      # Executable packaging
      {:burrito, "~> 1.0"},

      # Testing
      {:propcheck, "~> 1.4", only: :test},
      {:benchee, "~> 1.0", only: :dev},
      {:dialyxir, "~> 1.4", only: :dev, runtime: false},

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
