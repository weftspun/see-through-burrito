defmodule SeeThroughBurrito.Models do
  @moduledoc "Bumblebee model loading and caching"

  require Logger
  alias Bumblebee.Vision

  @doc "Load a diffusion model with Bumblebee"
  def load_diffusion(model_id, opts \\ []) do
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    File.mkdir_p!(cache_dir)

    Logger.info("Loading diffusion model: #{model_id}")

    Bumblebee.load_model({:hf, model_id}, cache_dir: cache_dir)
  end

  @doc "Load a VAE model"
  def load_vae(model_id, opts \\ []) do
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    File.mkdir_p!(cache_dir)

    Logger.info("Loading VAE model: #{model_id}")

    Bumblebee.load_model({:hf, model_id}, type: :autoencoder, cache_dir: cache_dir)
  end

  @doc "Load a CLIP text encoder"
  def load_text_encoder(model_id, opts \\ []) do
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    File.mkdir_p!(cache_dir)

    Logger.info("Loading text encoder: #{model_id}")

    Bumblebee.load_model({:hf, model_id}, type: :text, cache_dir: cache_dir)
  end

  @doc "Serve a model with Bumblebee and return a serving"
  def serve(model, featurizer, opts \\ []) do
    serving_opts = [
      batch_size: Keyword.get(opts, :batch_size, 1),
      device: :cuda
    ]

    Logger.info("Creating serving with options: #{inspect(serving_opts)}")

    Bumblebee.serve(model, featurizer, serving_opts)
  end

  @doc "Run inference on a batch"
  def run_inference(serving, input, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 60_000)

    case Bumblebee.run(serving, input, timeout: timeout) do
      {:ok, result} -> {:ok, result}
      {:error, reason} -> {:error, {:inference_failed, reason}}
    end
  end
end
