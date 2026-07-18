defmodule SeeThroughBurrito do
  @moduledoc """
  Elixir implementation of See-Through: anime illustration decomposition into semantic layers.

  This is a GPU-first implementation using Bumblebee/ExLA for neural inference,
  targeting deployment via Burrito for self-contained executables.
  """

  require Logger

  def version, do: "0.1.0"

  @doc "Process an anime illustration into semantic layers"
  @dialyzer {:nowarn_function, process: 2}
  def process(image_path, opts \\ []) do
    Logger.info("Loading image from #{image_path}")

    with {:ok, image} <- SeeThroughBurrito.Images.load(image_path),
         {:ok, preprocessed} <- SeeThroughBurrito.Pipeline.preprocess(image, opts),
         {:ok, vae_latents} <- SeeThroughBurrito.Encoder.encode_to_latents(preprocessed),
         {:ok, layers} <- SeeThroughBurrito.Layers.decompose(vae_latents, opts),
         {:ok, depth} <- SeeThroughBurrito.Depth.estimate(preprocessed, opts),
         {:ok, inpainted} <- SeeThroughBurrito.Inpaint.fill_holes(layers, opts) do
      {:ok, %{layers: inpainted, depth: depth}}
    else
      {:error, reason} ->
        Logger.error("Pipeline failed: #{inspect(reason)}")
        {:error, reason}
    end
  end
end
