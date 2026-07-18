defmodule SeeThroughBurrito.CLI do
  @moduledoc "Command-line interface for see-through processing"

  require Logger

  def main(args) do
    case OptionParser.parse(args,
      strict: [
        input: :string,
        output: :string,
        format: :string,
        steps: :integer,
        guidance: :float,
        depth_steps: :integer,
        width: :integer,
        height: :integer,
        seed: :integer,
        model_cache: :string,
        help: :boolean
      ],
      aliases: [i: :input, o: :output, f: :format, h: :help]
    ) do
      {opts, _positional, _invalid} ->
        if opts[:help] or is_nil(opts[:input]) do
          print_help()
          exit(0)
        else
          process(opts)
        end
    end
  end

  defp process(opts) do
    input_path = opts[:input]
    output_path = Keyword.get(opts, :output, "./output.svg")
    format = Keyword.get(opts, :format, "svg")

    Logger.info("Processing: #{input_path} -> #{output_path}")
    Logger.info("Format: #{format}")

    pipeline_opts = [
      steps: Keyword.get(opts, :steps, 30),
      guidance_scale: Keyword.get(opts, :guidance, 7.5),
      depth_steps: Keyword.get(opts, :depth_steps, 4),
      width: Keyword.get(opts, :width, 1024),
      height: Keyword.get(opts, :height, 1024),
      seed: Keyword.get(opts, :seed, 42),
      cache_dir: Keyword.get(opts, :model_cache, "/tmp/see-through-models")
    ]

    case SeeThroughBurrito.process(input_path, pipeline_opts) do
      {:ok, %{layers: layers, depth: depth}} ->
        case export_result(layers, depth, output_path, format) do
          {:ok, paths} ->
            Logger.info("✓ Successfully exported to #{output_path}")
            Enum.each(paths, fn path -> Logger.info("  - #{path}") end)
            exit(0)

          {:error, reason} ->
            Logger.error("Export failed: #{inspect(reason)}")
            exit(1)
        end

      {:error, reason} ->
        Logger.error("Processing failed: #{inspect(reason)}")
        exit(1)
    end
  end

  defp export_result(layers, depth, output_path, format) do
    output_dir = Path.dirname(output_path)
    File.mkdir_p!(output_dir)

    case format do
      "svg" ->
        with {:ok, svg_content} <- SeeThroughBurrito.SvgExport.to_svg(layers, depth),
             :ok <- File.write(output_path, svg_content),
             {:ok, manifest} <- SeeThroughBurrito.SvgExport.generate_manifest(layers, depth),
             :ok <- File.write(output_path <> ".json", manifest) do
          {:ok, [output_path, output_path <> ".json"]}
        else
          {:error, reason} -> {:error, reason}
        end

      _ ->
        {:error, {:unsupported_format, format}}
    end
  end

  defp print_help do
    IO.puts("""
    see-through-burrito: Anime illustration decomposition

    Usage:
      see-through-burrito --input <image> [OPTIONS]

    Options:
      -i, --input <path>           Input image path (required)
      -o, --output <path>          Output path (default: ./output.svg)
      -f, --format <fmt>           Output format: svg (default: svg)
      --steps <n>                  Diffusion steps (default: 30)
      --guidance <scale>           Guidance scale (default: 7.5)
      --depth-steps <n>            Depth estimation steps (default: 4)
      --width <px>                 Target width (default: 1024)
      --height <px>                Target height (default: 1024)
      --seed <n>                   Random seed (default: 42)
      --model-cache <dir>          Model cache directory
      -h, --help                   Show this help

    Examples:
      see-through-burrito -i anime.png -o layers.svg
      see-through-burrito -i anime.png --steps 50 --guidance 10.0
    """)
  end
end
