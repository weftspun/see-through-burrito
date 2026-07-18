defmodule MoxPipelineTest do
  use ExUnit.Case
  import Mox

  # Define mocks for behavior contracts
  defmock(MockModels, for: SeeThroughBurrito.ModelAdapter)
  defmock(MockEncoder, for: SeeThroughBurrito.EncoderAdapter)
  defmock(MockLayers, for: SeeThroughBurrito.LayerAdapter)
  defmock(MockDepth, for: SeeThroughBurrito.DepthAdapter)
  defmock(MockInpaint, for: SeeThroughBurrito.InpaintAdapter)
  defmock(MockImages, for: SeeThroughBurrito.ImageAdapter)

  setup :verify_on_exit!
  setup :set_mox_global

  test "mock image adapter contract" do
    expect(MockImages, :load, fn _path ->
      {:ok, %{width: 1024, height: 1024}}
    end)

    expect(MockImages, :to_rgb_tensor, fn _image ->
      {:ok, Nx.broadcast(0.5, {1024, 1024, 3})}
    end)

    # Verify mocks work as expected
    assert {:ok, image} = MockImages.load("anime.png")
    assert image.width == 1024

    assert {:ok, tensor} = MockImages.to_rgb_tensor(image)
    assert Nx.shape(tensor) == {1024, 1024, 3}
  end

  test "mock encoder adapter contract" do
    expect(MockEncoder, :encode_to_latents, fn image, _opts ->
      # Contract: encoder takes image, produces latent with 8x compression
      assert Nx.shape(image) == {1024, 1024, 3}
      {:ok, Nx.broadcast(0.1, {128, 128, 4})}
    end)

    image = Nx.broadcast(0.5, {1024, 1024, 3})
    assert {:ok, latents} = MockEncoder.encode_to_latents(image, [])
    assert Nx.shape(latents) == {128, 128, 4}
  end

  test "mock model adapter contract" do
    expect(MockModels, :run_inference, fn model, input ->
      # Contract: inference takes model descriptor and input tensor
      assert Map.has_key?(model, :id)
      assert is_tuple(input) or is_map(input)
      {:ok, Nx.broadcast(0.2, {1, 128, 128, 4})}
    end)

    model = %{id: "layerdiff-unet", type: :diffusion}
    latents = Nx.broadcast(0.5, {1, 128, 128, 4})
    embeddings = Nx.broadcast(0.1, {1, 77, 768})

    assert {:ok, noise_pred} = MockModels.run_inference(model, {latents, embeddings})
    assert Nx.shape(noise_pred) == {1, 128, 128, 4}
  end

  test "mock layers adapter contract" do
    expect(MockLayers, :decompose, fn latents, _opts ->
      # Contract: decompose latents into 24 semantic layers
      assert Nx.shape(latents) == {1, 128, 128, 4}

      layers = Enum.map(1..24, fn i ->
        %{name: "layer_#{i}", image: Nx.broadcast(Float.round(i / 24, 2), {512, 512, 3})}
      end)

      {:ok, layers}
    end)

    latents = Nx.broadcast(0.5, {1, 128, 128, 4})
    assert {:ok, layers} = MockLayers.decompose(latents, [])
    assert length(layers) == 24
    assert Enum.all?(layers, fn l -> Map.has_key?(l, :name) and Map.has_key?(l, :image) end)
  end

  test "mock depth adapter contract" do
    expect(MockDepth, :estimate, fn image, _opts ->
      # Contract: depth takes RGB image, produces single-channel depth map
      assert Nx.shape(image) == {1024, 1024, 3}
      {:ok, Nx.broadcast(0.5, {1024, 1024})}
    end)

    image = Nx.broadcast(0.5, {1024, 1024, 3})
    assert {:ok, depth} = MockDepth.estimate(image, [])
    assert Nx.shape(depth) == {1024, 1024}
  end

  test "mock inpaint adapter contract" do
    expect(MockInpaint, :fill_holes, fn layers, _opts ->
      # Contract: inpaint takes layers, returns same structure
      assert is_list(layers)
      assert length(layers) == 24
      {:ok, layers}
    end)

    layers = Enum.map(1..24, fn i ->
      %{name: "layer_#{i}", image: Nx.broadcast(0.5, {512, 512, 3})}
    end)

    assert {:ok, inpainted} = MockInpaint.fill_holes(layers, [])
    assert length(inpainted) == 24
  end
end
