defmodule SeeThroughBurrito.SvgExport do
  @moduledoc "Export decomposed layers as SVG with embedded base64 images"

  require Logger
  import XmlBuilder

  @doc "Create SVG document with layer stack"
  def to_svg(layers, depth_map, opts \\ []) do
    Logger.info("Generating SVG with #{length(layers)} layers")

    width = Keyword.get(opts, :width, 1024)
    height = Keyword.get(opts, :height, 1024)
    viewbox = "0 0 #{width} #{height}"

    svg_doc =
      element(:svg, [xmlns: "http://www.w3.org/2000/svg", viewBox: viewbox, width: width, height: height], [
        element(:defs, [], [
          # Embed depth map as pattern
          depth_pattern(depth_map),
          # Define layer styles
          element(:style, [], [text(layer_styles())])
        ]),
        # Add layers as g elements with embedded images
        layers
        |> Enum.with_index()
        |> Enum.map(&layer_group(&1, opts))
        |> Enum.intersperse(newline()),
        # Metadata
        element(:metadata, [], [
          element(:rdf, [
            "xmlns:rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
          ], [
            element("rdf:Description", [about: ""], [
              element(:"export:generatedBy", [], [text("SeeThroughBurrito")]),
              element(:"export:timestamp", [], [text(DateTime.utc_now() |> to_string())]),
              element(:"export:layerCount", [], [text(length(layers) |> to_string())])
            ])
          ])
        ])
      ])

    {:ok, to_string(svg_doc)}
  end

  defp layer_group({%{name: name, image: image}, idx}, opts) do
    opacity = Keyword.get(opts, :opacity, 1.0)
    b64_image = encode_image_b64(image)

    element(:g, [id: "layer-#{idx}", class: "layer", "data-name": name, opacity: "#{opacity}"], [
      element(:image, [
        href: "data:image/png;base64,#{b64_image}",
        width: "100%",
        height: "100%",
        preserveAspectRatio: "none"
      ], []),
      element(:title, [], [text(name)]),
      # Add metadata
      element(:metadata, [], [
        element(:"layer:name", [], [text(name)]),
        element(:"layer:index", [], [text(idx |> to_string())]),
        element(:"layer:type", [], [text("raster")])
      ])
    ])
  end

  defp depth_pattern(depth_map) do
    b64_depth = encode_image_b64(depth_map)

    element(:pattern, [id: "depth_pattern", width: "100%", height: "100%", patternUnits: "objectBoundingBox"], [
      element(:image, [href: "data:image/png;base64,#{b64_depth}", width: "100%", height: "100%"], [])
    ])
  end

  defp layer_styles() do
    """
    .layer {
      mix-blend-mode: normal;
    }
    .layer:hover {
      opacity: 0.8 !important;
    }
    """
  end

  @doc "Convert tensor to base64-encoded PNG"
  defp encode_image_b64(tensor) do
    # TODO: Implement PNG encoding
    # For now, return placeholder
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
  end

  @doc "Export layers to individual SVG files"
  def export_layers(layers, output_dir, opts \\ []) do
    File.mkdir_p!(output_dir)

    layers
    |> Enum.with_index()
    |> Enum.map(fn {layer, idx} -> export_single_layer(layer, output_dir, idx, opts) end)
    |> Enum.filter(&match?({:ok, _}, &1))
    |> case do
      results when length(results) == length(layers) ->
        {:ok, results}

      results ->
        {:warn, {:partial_export, results}}
    end
  end

  defp export_single_layer(%{name: name, image: image}, dir, idx, _opts) do
    filename = "#{idx}_#{String.replace(name, " ", "_")}.svg"
    path = Path.join(dir, filename)

    svg = single_layer_svg(image, name)

    case File.write(path, svg) do
      :ok -> {:ok, path}
      {:error, reason} -> {:error, {:write_failed, reason}}
    end
  end

  defp single_layer_svg(image, name) do
    b64 = encode_image_b64(image)

    svg_doc =
      element(:svg, [xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 1024 1024", width: 1024, height: 1024], [
        element(:image, [
          href: "data:image/png;base64,#{b64}",
          width: "100%",
          height: "100%"
        ], []),
        element(:title, [], [text(name)])
      ])

    to_string(svg_doc)
  end

  @doc "Generate layer manifest JSON"
  def generate_manifest(layers, depth_map, opts \\ []) do
    manifest = %{
      "format" => "svg",
      "version" => "1.0",
      "generated_at" => DateTime.utc_now() |> DateTime.to_iso8601(),
      "layer_count" => length(layers),
      "layers" =>
        layers
        |> Enum.with_index()
        |> Enum.map(fn {%{name: name}, idx} ->
          %{
            "index" => idx,
            "name" => name,
            "visible" => true,
            "opacity" => 1.0
          }
        end),
      "metadata" => %{
        "width" => Keyword.get(opts, :width, 1024),
        "height" => Keyword.get(opts, :height, 1024),
        "has_depth" => depth_map != nil,
        "output_format" => "svg"
      }
    }

    {:ok, Jason.encode!(manifest)}
  end
end
