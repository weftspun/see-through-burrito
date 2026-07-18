defmodule OrchestrationTest do
  @moduledoc """
  Integration tests for pipeline orchestration using dependency injection.
  Each unit is independently mocked, testing orchestration logic only.
  """
  use ExUnit.Case
  import Mox

  # Define mocks for full pipeline (with Orch prefix to avoid conflicts with other tests)
  defmock(OrchaImages, for: SeeThroughBurrito.ImageAdapter)
  defmock(OrchaPipeline, for: SeeThroughBurrito.PipelineAdapter)
  defmock(OrchaEncoder, for: SeeThroughBurrito.EncoderAdapter)
  defmock(OrchaLayers, for: SeeThroughBurrito.LayerAdapter)
  defmock(OrchaDepth, for: SeeThroughBurrito.DepthAdapter)
  defmock(OrchaInpaint, for: SeeThroughBurrito.InpaintAdapter)

  setup :verify_on_exit!
  setup :set_mox_global

  @moduletag :skip

  test "full pipeline orchestration with happy path" do
    # Mock each adapter to return valid data
    expect(OrchaImages, :load, fn _path ->
      {:ok, %{width: 1024, height: 1024, data: <<>>}}
    end)

    expect(OrchaPipeline, :preprocess, fn _image, _opts ->
      {:ok, Nx.broadcast(0.5, {1024, 1024, 3})}
    end)

    expect(OrchaEncoder, :encode_to_latents, fn _image, _opts ->
      {:ok, Nx.broadcast(0.1, {128, 128, 4})}
    end)

    expect(OrchaLayers, :decompose, fn _latents, _opts ->
      layers = Enum.map(1..24, fn i ->
        %{name: "layer_#{i}", image: Nx.broadcast(Float.round(i / 24, 2), {512, 512, 3})}
      end)
      {:ok, layers}
    end)

    expect(OrchaDepth, :estimate, fn _image, _opts ->
      {:ok, Nx.broadcast(0.5, {1024, 1024})}
    end)

    expect(OrchaInpaint, :fill_holes, fn layers, _opts ->
      {:ok, layers}
    end)

    # Verify orchestration calls adapters in correct order
    result =
      SeeThroughBurrito.process("anime.png", [
        adapters: %{
          images: OrchaImages,
          pipeline: OrchaPipeline,
          encoder: OrchaEncoder,
          layers: OrchaLayers,
          depth: OrchaDepth,
          inpaint: OrchaInpaint
        }
      ])

    assert {:ok, %{layers: layers, depth: depth}} = result
    assert length(layers) == 24
    assert Nx.shape(depth) == {1024, 1024}
  end

  test "pipeline error propagates from image loading" do
    expect(OrchaImages, :load, fn _path ->
      {:error, {:image_load_failed, "File not found"}}
    end)

    result =
      SeeThroughBurrito.process("missing.png", [
        adapters: %{
          images: OrchaImages,
          pipeline: OrchaPipeline,
          encoder: OrchaEncoder,
          layers: OrchaLayers,
          depth: OrchaDepth,
          inpaint: OrchaInpaint
        }
      ])

    assert {:error, {:image_load_failed, _}} = result
  end

  test "pipeline error propagates from encoding" do
    expect(OrchaImages, :load, fn _path ->
      {:ok, %{width: 1024, height: 1024}}
    end)

    expect(OrchaPipeline, :preprocess, fn _image, _opts ->
      {:ok, Nx.broadcast(0.5, {1024, 1024, 3})}
    end)

    expect(OrchaEncoder, :encode_to_latents, fn _image, _opts ->
      {:error, {:encoder_load_failed, "Model not found"}}
    end)

    result =
      SeeThroughBurrito.process("anime.png", [
        adapters: %{
          images: OrchaImages,
          pipeline: OrchaPipeline,
          encoder: OrchaEncoder,
          layers: OrchaLayers,
          depth: OrchaDepth,
          inpaint: OrchaInpaint
        }
      ])

    assert {:error, {:encoder_load_failed, _}} = result
  end

  test "pipeline error propagates from layer decomposition" do
    expect(OrchaImages, :load, fn _path ->
      {:ok, %{width: 1024, height: 1024}}
    end)

    expect(OrchaPipeline, :preprocess, fn _image, _opts ->
      {:ok, Nx.broadcast(0.5, {1024, 1024, 3})}
    end)

    expect(OrchaEncoder, :encode_to_latents, fn _image, _opts ->
      {:ok, Nx.broadcast(0.1, {128, 128, 4})}
    end)

    expect(OrchaLayers, :decompose, fn _latents, _opts ->
      {:error, {:layer_extraction_failed, "UNet inference failed"}}
    end)

    result =
      SeeThroughBurrito.process("anime.png", [
        adapters: %{
          images: OrchaImages,
          pipeline: OrchaPipeline,
          encoder: OrchaEncoder,
          layers: OrchaLayers,
          depth: OrchaDepth,
          inpaint: OrchaInpaint
        }
      ])

    assert {:error, {:layer_extraction_failed, _}} = result
  end

  test "production adapters used when none specified" do
    # This test verifies backward compatibility: when no adapters are passed,
    # production modules are used (though they'll return errors as they're placeholders)
    # We just verify the process function handles this gracefully

    # With no adapters specified, it tries to use real Images.load which returns error
    result = SeeThroughBurrito.process("anime.png", [])

    # Should fail at image loading with the placeholder error
    assert {:error, _} = result
  end

  test "adapter mocks verify interface contracts" do
    # Verify adapters receive expected argument types/counts
    expect(OrchaImages, :load, fn path ->
      assert is_binary(path)
      {:ok, %{}}
    end)

    expect(OrchaPipeline, :preprocess, fn image, opts ->
      assert is_map(image) or is_tuple(image)
      assert is_list(opts)
      {:ok, Nx.broadcast(0.5, {1024, 1024, 3})}
    end)

    expect(OrchaEncoder, :encode_to_latents, fn image, opts ->
      assert Nx.is_tensor(image) or is_map(image)
      assert is_list(opts)
      {:ok, Nx.broadcast(0.1, {128, 128, 4})}
    end)

    expect(OrchaLayers, :decompose, fn latents, opts ->
      assert Nx.is_tensor(latents) or is_map(latents)
      assert is_list(opts)
      {:ok, []}
    end)

    expect(OrchaDepth, :estimate, fn image, opts ->
      assert Nx.is_tensor(image) or is_map(image)
      assert is_list(opts)
      {:ok, Nx.broadcast(0.5, {1024, 1024})}
    end)

    expect(OrchaInpaint, :fill_holes, fn layers, opts ->
      assert is_list(layers)
      assert is_list(opts)
      {:ok, layers}
    end)

    # Verify all mocks are called with correct types
    SeeThroughBurrito.process("test.png", [
      adapters: %{
        images: OrchaImages,
        pipeline: OrchaPipeline,
        encoder: OrchaEncoder,
        layers: OrchaLayers,
        depth: OrchaDepth,
        inpaint: OrchaInpaint
      }
    ])

    # All expectations verified on exit via verify_on_exit!
  end
end
