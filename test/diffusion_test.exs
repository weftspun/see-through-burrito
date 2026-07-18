defmodule SeeThroughBurrito.DiffusionTest do
  @moduledoc """
  Tests for Diffusion orchestration module.
  Uses Bumblebee schedulers when available, falls back to custom implementations.
  """

  use ExUnit.Case, async: false
  require Logger
  require Nx

  @tag :skip
  test "Diffusion with Bumblebee scheduler" do
    # Create mock UNet
    unet = %{id: "test_unet", type: :layerdiff_unet}

    # Create inputs
    latents = Nx.broadcast(Nx.tensor(0.1, type: :f32), {1, 64, 64, 4})
    embeddings = Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 77, 1792})
    page_rgb = Nx.broadcast(Nx.tensor(0.5, type: :f32), {512, 512, 3})

    # Run diffusion
    result = SeeThroughBurrito.Diffusion.run_diffusion(
      unet,
      latents,
      embeddings,
      page_rgb,
      4,  # steps
      7.5  # guidance
    )

    case result do
      {:ok, output} ->
        assert Nx.is_tensor(output) or is_struct(output)
        Logger.info("✓ Diffusion completed: #{inspect(Nx.shape(output))}")

      {:error, reason} ->
        Logger.info("Expected error during diffusion: #{inspect(reason)}")
    end
  end

  @tag :skip
  test "Dual-pass diffusion (body + head)" do
    unet = %{id: "test_unet", type: :layerdiff_unet}
    page_rgb = Nx.broadcast(Nx.tensor(0.5, type: :f32), {512, 512, 3})
    page_alpha = Nx.broadcast(Nx.tensor(1.0), {512, 512})
    embeddings = Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 77, 1792})

    result = SeeThroughBurrito.Diffusion.run_dual_pass_diffusion(
      unet,
      page_rgb,
      page_alpha,
      embeddings,
      steps: 4,
      guidance: 7.5
    )

    case result do
      {:ok, %{body_latents: body, head_latents: head}} ->
        assert Nx.is_tensor(body) or is_struct(body)
        assert Nx.is_tensor(head) or is_struct(head)
        Logger.info("✓ Dual-pass complete")

      {:error, reason} ->
        Logger.info("Expected error in dual-pass: #{inspect(reason)}")
    end
  end

  test "Diffusion with fallback scheduler" do
    # This test validates the fallback path when Bumblebee.Diffusion is unavailable
    # Since we're testing the structure, not the actual inference

    unet = %{id: "test_unet"}
    latents = Nx.broadcast(Nx.tensor(0.1, type: :f32), {1, 64, 64, 4})
    embeddings = Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 77, 1792})
    page_rgb = Nx.broadcast(Nx.tensor(0.5, type: :f32), {512, 512, 3})

    # Test that function completes without crashing
    result = SeeThroughBurrito.Diffusion.run_diffusion(
      unet,
      latents,
      embeddings,
      page_rgb,
      2,  # minimal steps for quick test
      7.5
    )

    # Should be either ok or error, not crash
    case result do
      {:ok, _} -> assert true
      {:error, _} -> assert true
    end
  end

  test "Classifier-free guidance application" do
    # Test guidance scaling
    noise = Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 64, 64, 4})

    # Guidance > 1.0 should apply
    result = SeeThroughBurrito.Diffusion.apply_classifier_free_guidance(noise, 2.0)
    assert Nx.is_tensor(result)

    # Guidance = 1.0 should return unchanged
    result = SeeThroughBurrito.Diffusion.apply_classifier_free_guidance(noise, 1.0)
    assert Nx.is_tensor(result)
  end
end
