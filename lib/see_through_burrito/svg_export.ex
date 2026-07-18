defmodule SeeThroughBurrito.SvgExport do
  @moduledoc "Export decomposed layers as SVG with embedded base64 images"

  require Logger

  @doc "Create SVG document with layer stack"
  def to_svg(layers, depth_map, opts \\ []) do
    Logger.info("Generating SVG with #{length(layers)} layers")

    width = Keyword.get(opts, :width, 1024)
    height = Keyword.get(opts, :height, 1024)

    svg = build_svg(layers, depth_map, width, height)
    {:ok, svg}
  end

  defp build_svg(layers, depth_map, width, height) do
    layers_svg = layers |> Enum.with_index() |> Enum.map(&layer_svg(&1))
    styles = layer_styles()
    depth_pattern = encode_image_b64(depth_map)
    timestamp = DateTime.utc_now() |> DateTime.to_iso8601()

    """
    <?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 #{width} #{height}" width="#{width}" height="#{height}">
      <defs>
        <style>
          #{styles}
        </style>
        <pattern id="depth_pattern" width="100%" height="100%" patternUnits="objectBoundingBox">
          <image href="data:image/png;base64,#{depth_pattern}" width="100%" height="100%"/>
        </pattern>
      </defs>
      <g id="layers">
        #{Enum.join(layers_svg, "\n  ")}
      </g>
      <metadata>
        <rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                         xmlns:export="http://weftspun.art/see-through/"
                         export:generatedBy="SeeThroughBurrito"
                         export:timestamp="#{timestamp}"
                         export:layerCount="#{length(layers)}"/>
      </metadata>
    </svg>
    """
  end

  defp layer_svg({%{name: name, image: _image}, idx}) do
    b64_image = encode_image_b64(nil)

    """
    <g id="layer-#{idx}" class="layer" data-name="#{name}" opacity="1.0">
      <image href="data:image/png;base64,#{b64_image}" width="100%" height="100%" preserveAspectRatio="none"/>
      <title>#{name}</title>
      <metadata>
        <layer:name xmlns:layer="http://weftspun.art/see-through/">#{name}</layer:name>
        <layer:index>#{idx}</layer:index>
        <layer:type>raster</layer:type>
      </metadata>
    </g>
    """
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

  defp encode_image_b64(_tensor) do
    # Placeholder 1x1 transparent PNG
    # Placeholder 1x1 transparent PNG
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

  defp single_layer_svg(_image, name) do
    b64 = encode_image_b64(nil)

    """
    <?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">
      <image href="data:image/png;base64,#{b64}" width="100%" height="100%"/>
      <title>#{name}</title>
    </svg>
    """
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
