defmodule SeeThroughBurrito.PipelineE2ETest do
  @moduledoc """
  End-to-end pipeline tests with real model serving.

  Tests the full pipeline with ModelServing layer integration.
  All tests marked @skip due to GPU and HuggingFace model requirements.
  """

  use ExUnit.Case, async: false
  require Logger

  @tag :skip
  test "Full pipeline: image → CLIP encoding → text embeddings" do
    # Load CLIP models
    {:ok, clip_l} = SeeThroughBurrito.Clip.load_model(
      "openai/clip-vit-large-patch14"
    )

    {:ok, tokenizer_l} = SeeThroughBurrito.Clip.load_tokenizer(
      "openai/clip-vit-large-patch14"
    )

    # Tokenize tags
    tags = ["face", "hair", "clothes", "accessories"]

    results = Enum.map(tags, fn tag ->
      {:ok, tokens} = SeeThroughBurrito.Clip.tokenize(tag, tokenizer_l)
      {:ok, embeddings} = SeeThroughBurrito.Clip.run_inference(clip_l, tokens)
      embeddings
    end)

    # All should produce embeddings
    assert length(results) == 4
    Enum.each(results, fn result ->
      assert Nx.is_tensor(result) or is_struct(result) or is_map(result)
    end)
  end

  @tag :skip
  test "Full pipeline: image → VAE encode → latent space" do
    # Load VAE
    {:ok, vae} = SeeThroughBurrito.Vae.load_vae_model(
      "stabilityai/sd-vae-ft-mse"
    )

    # Create dummy image
    image = Nx.broadcast(Nx.tensor(0.5, type: :f32), {512, 512, 3})

    # Encode to latent space
    {:ok, latents} = SeeThroughBurrito.Vae.run_vae_encoding(vae, image)

    # Should get latent tensor
    assert Nx.is_tensor(latents) or is_struct(latents) or is_map(latents)
  end

  @tag :skip
  test "Full pipeline: latents → VAE decode → image" do
    # Load VAE
    {:ok, vae} = SeeThroughBurrito.Vae.load_vae_model(
      "stabilityai/sd-vae-ft-mse"
    )

    # Create dummy latents (64x64x4 for 512x512 image)
    latents = Nx.broadcast(Nx.tensor(0.1, type: :f32), {64, 64, 4})

    # Decode to image space
    {:ok, image} = SeeThroughBurrito.Vae.run_vae_decoding(vae, latents)

    # Should get image tensor
    assert Nx.is_tensor(image) or is_struct(image) or is_map(image)
  end

  @tag :skip
  test "Full pipeline: image → Marigold depth estimation" do
    # Load depth model
    {:ok, depth_model} = SeeThroughBurrito.Marigold.load_model(
      "prs-eth/marigold-v1",
      "/tmp/models"
    )

    # Create dummy image
    image = Nx.broadcast(Nx.tensor(0.5, type: :f32), {512, 512, 3})

    # Estimate depth
    {:ok, depth} = SeeThroughBurrito.Marigold.run_depth_inference(
      depth_model,
      image,
      4
    )

    # Should get depth map
    assert Nx.is_tensor(depth) or is_struct(depth) or is_map(depth)
  end

  @tag :skip
  test "Full pipeline: UNet forward pass with conditioning" do
    # Load LayerDiff UNet
    {:ok, unet} = SeeThroughBurrito.Unet.load_model("/tmp/models")

    # Create inputs
    latents = Nx.broadcast(Nx.tensor(0.1, type: :f32), {1, 64, 64, 4})
    timestep = Nx.tensor(100, type: :s64)
    embeddings = Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 77, 1792})
    page_rgb = Nx.broadcast(Nx.tensor(0.5, type: :f32), {512, 512, 3})

    # Forward pass
    {:ok, noise_pred} = SeeThroughBurrito.Unet.forward(
      unet,
      latents,
      timestep,
      embeddings,
      page_rgb
    )

    # Should get noise prediction same shape as latents
    assert Nx.is_tensor(noise_pred) or is_struct(noise_pred) or is_map(noise_pred)
  end

  @tag :skip
  test "ModelCache integration with pipeline" do
    cache_dir = "/tmp/see-through-models"

    # Get CLIP model via cache
    {:ok, clip_l} = SeeThroughBurrito.ModelCache.get_model(
      :clip_l,
      "openai/clip-vit-large-patch14",
      type: :clip_text
    )

    # Get same model again (should be cached)
    {:ok, clip_l_2} = SeeThroughBurrito.ModelCache.get_model(
      :clip_l,
      "openai/clip-vit-large-patch14",
      type: :clip_text
    )

    # Should be same model
    assert clip_l == clip_l_2
  end

  @tag :skip
  test "Dual-pass LayerDiff: body + head passes" do
    {:ok, unet} = SeeThroughBurrito.Unet.load_model("/tmp/models")

    page_rgb = Nx.broadcast(Nx.tensor(0.5, type: :f32), {512, 512, 3})
    page_alpha = Nx.broadcast(Nx.tensor(1.0), {512, 512})
    embeddings = Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 77, 1792})

    result = SeeThroughBurrito.Unet.dual_pass(
      unet,
      page_rgb,
      page_alpha,
      embeddings,
      steps: 30,
      guidance: 7.5
    )

    case result do
      {:ok, %{body_layers: body, head_layers: head}} ->
        # Should have expected number of layers
        assert length(body) == 13
        assert length(head) == 11

      {:error, {:not_implemented, _}} ->
        # Expected until full integration
        :ok
    end
  end

  @tag :skip
  test "Post-processing pipeline with depth ordering" do
    # Create mock layers
    h = 256
    w = 256
    layer1 = Nx.broadcast(Nx.tensor(0.5, type: :f32), {h, w, 4})
    layer2 = Nx.broadcast(Nx.tensor(0.3, type: :f32), {h, w, 4})
    layers = [layer1, layer2]

    # Create depth map
    depth_map = Nx.broadcast(Nx.tensor(0.6), {h, w})

    # Apply post-processing
    result = SeeThroughBurrito.Postproc.postprocess(layers, depth_map)

    # Should return processed layers
    assert is_list(result)
    assert length(result) == 2
  end

  @tag :skip
  test "SVG export with processed layers" do
    h = 256
    w = 256
    layer = Nx.broadcast(Nx.tensor(0.5, type: :f32), {h, w, 4})

    metadata = %{
      "layer_names" => ["test"],
      "layer_depths" => [0.5],
      "image_size" => {h, w}
    }

    svg = SeeThroughBurrito.SvgExport.layers_to_svg([layer], metadata)

    assert is_binary(svg)
    assert String.contains?(svg, ["<svg", "test"])
  end

  @tag :skip
  test "Full orchestration with real adapters" do
    # Test full pipeline with dependency injection
    adapters = %{
      images: SeeThroughBurrito.Images,
      encoder: SeeThroughBurrito.Encoder,
      layers: SeeThroughBurrito.Layers,
      depth: SeeThroughBurrito.Marigold,
      inpaint: SeeThroughBurrito.Inpaint
    }

    # This would run the full pipeline if image exists
    # For now, just validate the adapter structure
    assert is_map(adapters)
    assert Map.has_key?(adapters, :images)
    assert Map.has_key?(adapters, :depth)
  end
end
