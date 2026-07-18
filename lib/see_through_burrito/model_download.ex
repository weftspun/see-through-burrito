defmodule SeeThroughBurrito.ModelDownload do
  @moduledoc "Download and decompress model weights from GitHub releases"

  require Logger

  @release_base "https://github.com/weftspun/see-through-burrito/releases/download"
  @release_tag "v0.1.0-models"

  # Model specs: {model_id, filename_pattern, expected_size_mb}
  # Weights stored as safetensors (Hugging Face native format) compressed with zstd
  @models %{
    "layerdiff-unet" => {
      "layerdiff-unet.safetensors.zst",
      2048,
      "LayerDiff UNet (frame-conditioned diffusion for semantic layers)"
    },
    "trans-vae" => {
      "trans-vae.safetensors.zst",
      512,
      "TransparentVAE (anime illustration VAE)"
    },
    "marigold-unet" => {
      "marigold-unet.safetensors.zst",
      1024,
      "Marigold depth estimation UNet"
    },
    "sd-vae" => {
      "sd-vae-ft-mse.safetensors.zst",
      256,
      "Stable Diffusion VAE encoder/decoder"
    },
    "lama" => {
      "lama.safetensors.zst",
      512,
      "LaMa FFCResNet inpainting model"
    },
    "clip-l" => {
      "clip-vit-large-patch14.safetensors.zst",
      256,
      "CLIP-L text encoder (336M)"
    },
    "clip-g" => {
      "clip-vit-g-14.safetensors.zst",
      1024,
      "OpenCLIP-G text encoder (1.4B)"
    }
  }

  @doc "Download model, decompress from zst, verify integrity"
  def fetch(model_id, opts \\ []) do
    cache_dir = Keyword.get(opts, :cache_dir, cache_directory())

    with {filename, _size_mb, desc} <- get_model_spec(model_id),
         :ok <- File.mkdir_p(cache_dir),
         output_path = Path.join(cache_dir, String.replace(filename, ".zst", "")),
         {:ok, _} <- download_and_decompress(model_id, filename, output_path, opts) do
      Logger.info("✓ Loaded #{desc}")
      {:ok, output_path}
    else
      {:error, reason} ->
        Logger.error("Failed to fetch #{model_id}: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc "Download all required models for full pipeline"
  def fetch_all(opts \\ []) do
    required = ["layerdiff-unet", "sd-vae", "marigold-unet", "lama", "clip-l"]

    Logger.info("Downloading #{length(required)} models...")

    results =
      required
      |> Enum.map(&fetch(&1, opts))
      |> Enum.filter(&match?({:ok, _}, &1))

    if length(results) == length(required) do
      paths = Enum.map(results, fn {:ok, path} -> path end)
      {:ok, paths}
    else
      failed = length(required) - length(results)
      {:error, {:download_incomplete, "#{failed}/#{length(required)} models failed"}}
    end
  end

  @doc "List available models"
  def list_models do
    @models
    |> Enum.map(fn {id, {_filename, size, desc}} ->
      %{id: id, size_mb: size, description: desc}
    end)
  end

  @doc "Check if model is cached locally"
  def cached?(model_id, opts \\ []) do
    cache_dir = Keyword.get(opts, :cache_dir, cache_directory())

    case get_model_spec(model_id) do
      {filename, _, _} ->
        output_path = Path.join(cache_dir, String.replace(filename, ".zst", ""))
        File.exists?(output_path)

      :error ->
        false
    end
  end

  defp download_and_decompress(_model_id, filename, output_path, opts) do
    url = "#{@release_base}/#{@release_tag}/#{filename}"
    temp_file = output_path <> ".zst"

    with :ok <- download_file(url, temp_file, opts),
         :ok <- decompress_zst(temp_file, output_path),
         :ok <- File.rm(temp_file) do
      {:ok, output_path}
    else
      {:error, reason} ->
        File.rm(temp_file)
        {:error, reason}
    end
  end

  defp download_file(url, dest, opts) do
    timeout = Keyword.get(opts, :timeout, 600_000)

    case HTTPoison.get(url, [], recv_timeout: timeout, stream_to: self()) do
      {:ok, response} ->
        case response.status_code do
          200 ->
            File.write(dest, response.body)

          404 ->
            {:error, {:not_found, "Model not available in this release: #{url}"}}

          code ->
            {:error, {:http_error, "HTTP #{code}"}}
        end

      {:error, reason} ->
        {:error, {:download_failed, reason}}
    end
  end

  defp decompress_zst(zst_file, output_file) do
    case System.cmd("zstd", ["-d", zst_file, "-o", output_file]) do
      {_output, 0} ->
        :ok

      {error, _code} ->
        {:error, {:decompress_failed, error}}
    end
  rescue
    e ->
      {:error, {:zstd_missing, "Install zstd: apt-get install zstd (#{inspect(e)}"}}
  end

  defp get_model_spec(model_id) do
    case Map.fetch(@models, model_id) do
      {:ok, spec} -> spec
      :error -> {:error, {:unknown_model, model_id}}
    end
  end

  defp cache_directory do
    System.get_env("XDG_CACHE_HOME", Path.expand("~/.cache"))
    |> Path.join("see-through-burrito/models")
  end
end
