defmodule SeeThroughBurrito.Models do
  @moduledoc "Model loading and inference wrapper"

  require Logger

  @doc "Load a model from release-distributed safetensors file"
  def load_model(model_id, opts \\ []) do
    Logger.info("Loading model: #{model_id}")

    with {:ok, model_path} <- SeeThroughBurrito.ModelDownload.fetch(model_id, opts) do
      Logger.debug("Model cached at: #{model_path}")
      {:ok, %{id: model_id, path: model_path, type: :safetensors}}
    else
      {:error, reason} ->
        Logger.error("Failed to load model #{model_id}: #{inspect(reason)}")
        {:error, {:model_load_failed, reason}}
    end
  end

  @doc "Load diffusion model (UNet)"
  def load_diffusion(model_id, opts \\ []) do
    load_model(model_id, opts)
  end

  @doc "Load VAE (encoder/decoder)"
  def load_vae(model_id, opts \\ []) do
    load_model(model_id, opts)
  end

  @doc "Load text encoder (CLIP)"
  def load_text_encoder(model_id, opts \\ []) do
    load_model(model_id, opts)
  end

  @doc "Check if model is available locally"
  def cached?(model_id, opts \\ []) do
    SeeThroughBurrito.ModelDownload.cached?(model_id, opts)
  end

  @doc "Placeholder for future Bumblebee integration"
  def run_inference(_model, _input, _opts \\ []) do
    # TODO: Integrate with Bumblebee 0.7.0 when Axon models are wired
    Logger.warning("Model inference not yet implemented - awaiting Bumblebee integration")
    {:error, {:not_implemented, "Inference requires Bumblebee Axon model wiring"}}
  end
end
