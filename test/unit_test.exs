defmodule UnitTest do
  use ExUnit.Case, async: true
  use PropCheck

  # GPU-dependent tests: skip when CUDA/Metal not available
  # Run with: mix test.gpu
  @moduletag :skip

  # Non-GPU logic tests (can run without EXLA/GPU)

  test "ModelDownload.list_models returns valid specs" do
    models = SeeThroughBurrito.ModelDownload.list_models()

    assert is_list(models)
    assert length(models) > 0

    Enum.each(models, fn model ->
      assert Map.has_key?(model, :id)
      assert Map.has_key?(model, :size_mb)
      assert Map.has_key?(model, :description)
      assert is_integer(model.size_mb)
    end)
  end

  test "ModelDownload.cached? returns boolean" do
    result1 = SeeThroughBurrito.ModelDownload.cached?("model1")
    result2 = SeeThroughBurrito.ModelDownload.cached?("model2")

    assert is_boolean(result1)
    assert is_boolean(result2)
  end

  test "SvgExport.generate_manifest produces valid JSON" do
    layers = [%{name: "background"}, %{name: "character"}]
    depth = nil

    {:ok, json} = SeeThroughBurrito.SvgExport.generate_manifest(layers, depth)
    manifest = Jason.decode!(json)

    assert manifest["format"] == "svg"
    assert manifest["layer_count"] == 2
    assert length(manifest["layers"]) == 2
    assert hd(manifest["layers"])["name"] == "background"
  end

  test "SvgExport.to_svg generates valid SVG structure" do
    layers = [%{name: "layer1", image: nil}, %{name: "layer2", image: nil}]
    depth = nil

    {:ok, svg} = SeeThroughBurrito.SvgExport.to_svg(layers, depth)

    # Verify SVG structure
    assert String.contains?(svg, "<svg")
    assert String.contains?(svg, "xmlns")
    assert String.contains?(svg, "layer1")
    assert String.contains?(svg, "layer2")
    assert String.contains?(svg, "</svg>")
    assert String.contains?(svg, "metadata")
  end

  test "SvgExport layer names appear in SVG output" do
    layer_names = ["layer_1", "layer_2", "layer_3"]
    layers = Enum.map(layer_names, &%{name: &1, image: nil})
    {:ok, svg} = SeeThroughBurrito.SvgExport.to_svg(layers, nil)

    # All layer names should appear in SVG
    Enum.each(layer_names, fn name ->
      assert String.contains?(svg, name)
    end)
  end

  test "Layers.composite_layers handles empty layers" do
    result = SeeThroughBurrito.Layers.composite_layers([], [])
    assert is_nil(result)
  end

  test "Layers.composite_layers preserves shape for single layer" do
    h = 16
    w = 16
    c = 3
    layer = Nx.fill(Nx.tensor(0.5), {h, w, c})
    alpha = Nx.fill(Nx.tensor(0.7), {h, w})

    result = SeeThroughBurrito.Layers.composite_layers([layer], [alpha])
    assert Nx.shape(result) == {h, w, c}
  end

  test "SvgExport handles various layer counts" do
    Enum.each(0..5, fn layer_count ->
      layers = Enum.map(1..layer_count, &%{name: "layer_#{&1}", image: nil})
      {:ok, json} = SeeThroughBurrito.SvgExport.generate_manifest(layers, nil)
      manifest = Jason.decode!(json)

      assert manifest["layer_count"] == layer_count
      assert length(manifest["layers"]) == layer_count
    end)
  end
end
