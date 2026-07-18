defmodule SeeThroughBurrito.Pipeline do
  @moduledoc "Main processing pipeline orchestration"

  require Logger
  import Nx.Defn

  @doc "Preprocess image for model input"
  def preprocess(image, opts \\ []) do
    target_height = Keyword.get(opts, :height, 1024)
    target_width = Keyword.get(opts, :width, 1024)

    Logger.info("Preprocessing image to #{target_width}x#{target_height}")

    with {:ok, rgb_tensor} <- SeeThroughBurrito.Images.to_rgb_tensor(image),
         padded = SeeThroughBurrito.Images.pad_to_8_divisible(rgb_tensor),
         normalized = normalize_sdxl(padded) do
      {:ok, normalized}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc "Normalize using SDXL standards"
  defn normalize_sdxl(tensor) do
    mean = Nx.tensor([0.485, 0.456, 0.406])
    std = Nx.tensor([0.229, 0.224, 0.225])

    tensor
    |> Nx.subtract(mean)
    |> Nx.divide(std)
  end

  @doc "Encode prompt text to embeddings"
  def encode_prompt(prompt, _tokenizer, text_encoder) do
    Logger.debug("Encoding prompt: #{prompt}")

    case SeeThroughBurrito.Models.run_inference(text_encoder, prompt) do
      {:ok, embeddings} -> {:ok, embeddings}
      {:error, reason} -> {:error, {:prompt_encoding_failed, reason}}
    end
  end

  @doc "Run diffusion model to generate layer masks"
  def run_diffusion(unet, latents, embeddings, opts \\ []) do
    steps = Keyword.get(opts, :steps, 30)
    guidance_scale = Keyword.get(opts, :guidance_scale, 7.5)

    Logger.info("Running diffusion with #{steps} steps, guidance #{guidance_scale}")

    diffuse_loop(unet, latents, embeddings, steps, guidance_scale, [])
  end

  defp diffuse_loop(_unet, _latents, _embeddings, 0, _guidance, acc) do
    {:ok, Enum.reverse(acc)}
  end

  defp diffuse_loop(unet, latents, embeddings, step, guidance, acc) do
    Logger.debug("Diffusion step #{step}")

    case SeeThroughBurrito.Models.run_inference(unet, {latents, embeddings}) do
      {:ok, noise_pred} ->
        guided = Nx.multiply(noise_pred, guidance)
        new_latents = Nx.subtract(latents, guided)
        diffuse_loop(unet, new_latents, embeddings, step - 1, guidance, [new_latents | acc])

      {:error, reason} ->
        {:error, {:diffusion_step_failed, reason}}
    end
  end

end
