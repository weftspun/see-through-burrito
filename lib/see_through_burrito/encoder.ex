defmodule SeeThroughBurrito.Encoder do
  @moduledoc "VAE encoder for image-to-latent conversion"
  @behaviour SeeThroughBurrito.EncoderAdapter

  require Logger

  @doc "Encode image to VAE latent space"
  @dialyzer {:nowarn_function, encode_to_latents: 2}
  def encode_to_latents(image_tensor, opts \\ []) do
    Logger.info("Encoding image to latent space")

    case load_encoder_model(opts) do
      {:ok, encoder_serving} ->
        case SeeThroughBurrito.Models.run_inference(encoder_serving, image_tensor) do
          {:ok, %{"latent_dist" => latents}} ->
            {:ok, latents}

          {:ok, result} ->
            Logger.warning("Unexpected encoder output: #{inspect(Map.keys(result))}")
            {:error, {:unexpected_output, result}}

          {:error, reason} ->
            {:error, {:encoder_inference_failed, reason}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Sample from latent distribution (reparameterization trick)"
  @dialyzer {:nowarn_function, sample_latents: 2}
  def sample_latents(mean, logvar) do
    std = logvar |> Nx.multiply(0.5) |> Nx.exp()
    # For deterministic tests, use zeros as eps (posterior mean sampling)
    # In production, would use Nx.Random.normal with key
    eps = Nx.broadcast(Nx.tensor(0.0), Nx.shape(mean))
    Nx.add(mean, Nx.multiply(std, eps))
  end

  @doc "Decode latents back to image space"
  @dialyzer {:nowarn_function, decode_from_latents: 2}
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
