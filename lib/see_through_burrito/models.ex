defmodule SeeThroughBurrito.Models do
  @moduledoc "Bumblebee model loading and caching"

  require Logger
  alias Bumblebee.Vision

  @doc "Load a diffusion model from GGUF weights (release distribution)"
  def load_diffusion(model_id, opts \\ []) do
    Logger.info("Loading diffusion model: #{model_id}")

    with {:ok, model_path} <- SeeThroughBurrito.ModelDownload.fetch(model_id, opts),
         {:ok, model} <- load_gguf_model(model_path, opts) do
      {:ok, model}
    else
      {:error, reason} -> {:error, {:diffusion_load_failed, reason}}
    end
  end

  @doc "Load a VAE model from GGUF weights"
  def load_vae(model_id, opts \\ []) do
    Logger.info("Loading VAE model: #{model_id}")

    with {:ok, model_path} <- SeeThroughBurrito.ModelDownload.fetch(model_id, opts),
         {:ok, model} <- load_gguf_model(model_path, opts) do
      {:ok, model}
    else
      {:error, reason} -> {:error, {:vae_load_failed, reason}}
    end
  end

  @doc "Load a CLIP text encoder from GGUF weights"
  def load_text_encoder(model_id, opts \\ []) do
    Logger.info("Loading text encoder: #{model_id}")

    with {:ok, model_path} <- SeeThroughBurrito.ModelDownload.fetch(model_id, opts),
         {:ok, model} <- load_gguf_model(model_path, opts) do
      {:ok, model}
    else
      {:error, reason} -> {:error, {:text_encoder_load_failed, reason}}
    end
  end

  defp load_gguf_model(path, _opts) do
    # Load safetensors weights via Bumblebee
    Logger.debug("Loading safetensors model from: #{path}")

    case File.exists?(path) do
      true ->
        # Bumblebee.load_model expects {:file, path} for local safetensors
        case Bumblebee.load_model({:file, path}) do
          {:ok, model} -> {:ok, model}
          {:error, reason} -> {:error, {:safetensors_load_failed, reason}}
        end

      false ->
        {:error, {:model_not_found, path}}
    end
  rescue
    e -> {:error, {:load_safetensors_failed, inspect(e)}}
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
