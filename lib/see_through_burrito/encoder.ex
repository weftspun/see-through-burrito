defmodule SeeThroughBurrito.Encoder do
  @moduledoc "VAE encoder for image-to-latent conversion"

  require Logger
  import Nx.Defn

  @doc "Encode image to VAE latent space"
  def encode_to_latents(image_tensor, opts \\ []) do
    Logger.info("Encoding image to latent space")

    case load_encoder_model(opts) do
      {:ok, encoder_serving} ->
        case SeeThroughBurrito.Models.run_inference(encoder_serving, image_tensor) do
          {:ok, %{"latent_dist" => latents}} ->
            {:ok, latents}

          {:ok, result} ->
            Logger.warn("Unexpected encoder output: #{inspect(Map.keys(result))}")
            {:error, {:unexpected_output, result}}

          {:error, reason} ->
            {:error, {:encoder_inference_failed, reason}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Sample from latent distribution"
  defn sample_latents(mean, logvar, key) do
    std = Nx.exp(Nx.multiply(logvar, 0.5))
    eps = random_normal(Nx.shape(mean), key)
    Nx.add(mean, Nx.multiply(std, eps))
  end

  defn random_normal(shape, key) do
    # Placeholder for proper random sampling
    Nx.random_normal(shape, key: key)
  end

  @doc "Decode latents back to image space"
  def decode_from_latents(latents, opts \\ []) do
    Logger.info("Decoding latents to image space")

    case load_decoder_model(opts) do
      {:ok, decoder_serving} ->
        case SeeThroughBurrito.Models.run_inference(decoder_serving, latents) do
          {:ok, image} ->
            {:ok, image}

          {:error, reason} ->
            {:error, {:decoder_inference_failed, reason}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp load_encoder_model(opts) do
    model_id = Keyword.get(opts, :encoder_model, "stabilityai/sd-vae-ft-mse")
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    case SeeThroughBurrito.Models.load_vae(model_id, cache_dir: cache_dir) do
      {:ok, model} ->
        {:ok, model}

      {:error, reason} ->
        Logger.error("Failed to load encoder: #{inspect(reason)}")
        {:error, {:encoder_load_failed, reason}}
    end
  end

  defp load_decoder_model(opts) do
    model_id = Keyword.get(opts, :decoder_model, "stabilityai/sd-vae-ft-mse")
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    case SeeThroughBurrito.Models.load_vae(model_id, cache_dir: cache_dir) do
      {:ok, model} ->
        {:ok, model}

      {:error, reason} ->
        Logger.error("Failed to load decoder: #{inspect(reason)}")
        {:error, {:decoder_load_failed, reason}}
    end
  end
end
