defmodule SeeThroughBurrito.Unet do
  @moduledoc """
  LayerDiff UNet for semantic layer decomposition.
  Ported from see-through-cpp/src/unet_frame.cpp

  Provides:
  - Frame-conditional UNet inference
  - Dual-pass diffusion (body + head tags)
  - Noise prediction for diffusion scheduler
  """

  require Logger

  @doc """
  Load LayerDiff UNet model.

  The model is frame-conditional:
  - Input: (latents, timestep, embeddings, page_rgb, page_alpha)
  - Output: noise prediction (same shape as latents)

  From: see-through-cpp/src/unet_frame.cpp
  """
  def load_model(cache_dir) do
    Logger.info("Loading LayerDiff UNet model")

    model_id = "shitagaki-lab/layerdiff-unet"

    # Use ModelServing layer for consistent Bumblebee API handling
    SeeThroughBurrito.ModelServing.load_model(model_id, :unet, cache_dir: cache_dir)
  end

  @doc """
  Run LayerDiff UNet inference (noise prediction).

  Inputs:
  - latents: {batch, height/8, width/8, 4}
  - timestep: scalar or tensor
  - embeddings: {batch, 77, embedding_dim}
  - page_rgb: {height, width, 3} - conditioning image
  - page_alpha: {height, width} - conditioning mask (optional)

  Output:
  - noise_pred: {batch, height/8, width/8, 4} - predicted noise
  """
  def forward(model, latents, timestep, embeddings, page_rgb, _page_alpha \\ nil) do
    Logger.debug("Running LayerDiff UNet forward pass")

    case model do
      nil ->
        {:error, {:invalid_model, "Model is nil"}}

      _model ->
        # Prepare input: pack all conditioning info
        input = {latents, timestep, embeddings, page_rgb}
        SeeThroughBurrito.ModelServing.run_inference(model, input)
    end
  end

  @doc """
  Run dual-pass LayerDiff: body tags + head tags.

  Body pass: extract 13 body part layers
  Head pass: extract 11 head part layers

  From: see-through-cpp/src/see_through.cpp (lines 93-100)
  """
  @dialyzer {:nowarn_function, dual_pass: 5}
  def dual_pass(model, page_rgb, page_alpha, embeddings, opts \\ []) do
    Logger.info("Running dual-pass LayerDiff (body + head)")

    steps = Keyword.get(opts, :steps, 30)
    guidance = Keyword.get(opts, :guidance, 7.5)

    # Body pass (group 0)
    with {:ok, body_layers} <- pass(model, page_rgb, page_alpha, embeddings, 0, steps, guidance) do
      # Head pass (group 1)
      with {:ok, head_layers} <- pass(model, page_rgb, page_alpha, embeddings, 1, steps, guidance) do
        {:ok, %{body_layers: body_layers, head_layers: head_layers}}
      else
        {:error, reason} -> {:error, {:head_pass_failed, reason}}
      end
    else
      {:error, reason} -> {:error, {:body_pass_failed, reason}}
    end
  end

  @doc """
  Single LayerDiff pass (body or head).

  Args:
  - group_index: 0 = body (13 tags), 1 = head (11 tags)
  - steps: number of diffusion steps
  - guidance: classifier-free guidance scale
  """
  def pass(_model, _page_rgb, _page_alpha, _embeddings, _group_index, _steps, _guidance) do
    Logger.debug("Running LayerDiff pass")

    # Placeholder: would iterate through diffusion steps
    # For now, return error until Bumblebee wired
    {:error, {:not_implemented, "LayerDiff pass awaits diffusion scheduler integration"}}
  end

  @doc """
  Reshape LayerDiff output to layers.

  Model outputs: {batch, H, W, 4} where batch includes all 13/11 tags
  Returns: list of {tag_name, rgba_image} pairs
  """
  def outputs_to_layers(model_output, group_index) do
    Logger.debug("Converting UNet output to layers (group #{group_index})")

    # Placeholder
    {:error, :not_implemented}
  end

  @doc """
  Get the tag vocabulary for LayerDiff.

  Body tags (13): head-like labels for body parts
  Head tags (11): detailed head features

  From: see-through-cpp/src/pipeline.h
  """
  def body_tags do
    [
      "face",
      "eyes",
      "nose",
      "mouth",
      "hair_front",
      "hair_back",
      "hair_sides",
      "hair_bangs",
      "neck",
      "clothes_upper",
      "clothes_lower",
      "accessories",
      "background"
    ]
  end

  def head_tags do
    [
      "skin",
      "eyebrows",
      "hair_accessories",
      "gloves",
      "shoes",
      "socks",
      "clothes_outerwear",
      "accessories_head",
      "accessories_neck",
      "accessories_hand",
      "accessories_waist"
    ]
  end

  @doc """
  Full semantic layer list (24 total).
  """
  def all_layers do
    body_tags() ++ head_tags()
  end

  @doc """
  Get UNet output shape for given spatial dimensions.

  UNet operates on latent space: H/8 × W/8 × 4 channels
  Output shape same as input.
  """
  def get_output_shape(image_height, image_width) do
    latent_h = div(image_height, 8)
    latent_w = div(image_width, 8)
    {latent_h, latent_w, 4}
  end
end
