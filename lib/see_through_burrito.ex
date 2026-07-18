defmodule SeeThroughBurrito do
  @moduledoc """
  Elixir implementation of See-Through: anime illustration decomposition into semantic layers.

  This is a GPU-first implementation using Bumblebee/ExLA for neural inference,
  targeting deployment via Burrito for self-contained executables.
  """

  require Logger

  def version, do: "0.1.0"

  @doc "Process an anime illustration into semantic layers"
  @doc """
  Accepts optional adapters via :adapters key in opts for dependency injection.

  Example (testing with mocks):
    process(image_path, adapters: %{
      images: MockImages,
      encoder: MockEncoder,
      layers: MockLayers,
      depth: MockDepth,
      inpaint: MockInpaint
    })
  """
  @dialyzer {:nowarn_function, process: 2}
  def process(image_path, opts \\ []) do
    Logger.info("Loading image from #{image_path}")

    # Dependency injection with defaults
    adapters = Keyword.get(opts, :adapters, default_adapters())
    pipeline_opts = Keyword.delete(opts, :adapters)

    with {:ok, image} <- adapters.images.load(image_path),
         {:ok, preprocessed} <- adapters.pipeline.preprocess(image, pipeline_opts),
         {:ok, vae_latents} <- adapters.encoder.encode_to_latents(preprocessed, pipeline_opts),
         {:ok, layers} <- adapters.layers.decompose(vae_latents, pipeline_opts),
         {:ok, depth} <- adapters.depth.estimate(preprocessed, pipeline_opts),
         {:ok, inpainted} <- adapters.inpaint.fill_holes(layers, pipeline_opts) do
      {:ok, %{layers: inpainted, depth: depth}}
    else
      {:error, reason} ->
        Logger.error("Pipeline failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc "Default production adapters"
  defp default_adapters do
    %{
      images: SeeThroughBurrito.Images,
      pipeline: SeeThroughBurrito.Pipeline,
      encoder: SeeThroughBurrito.Encoder,
      layers: SeeThroughBurrito.Layers,
      depth: SeeThroughBurrito.Depth,
      inpaint: SeeThroughBurrito.Inpaint
    }
  end
end
