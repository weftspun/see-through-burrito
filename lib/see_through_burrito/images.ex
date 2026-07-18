defmodule SeeThroughBurrito.Images do
  @moduledoc "Image loading, preprocessing, and manipulation"

  require Logger

  @doc "Load an image from file as RGB tensor (0-1 range)"
  def load(path) do
    Logger.info("Loading image: #{path}")

    case Image.open(path) do
      {:ok, image} ->
        to_rgb_tensor(image)

      {:error, reason} ->
        Logger.error("Failed to load image #{path}: #{inspect(reason)}")
        {:error, {:image_load_failed, reason}}
    end
  end

  @doc "Convert image to RGB tensor (0-1 range)"
  def to_rgb_tensor(image) do
    try do
      # Image library returns Vix image, convert via as_tensor or binary
      width = Image.width(image)
      height = Image.height(image)

      # Use Image.write to get binary data
      case Image.write(image, suffix: ".ppm") do
        {:ok, binary} ->
          # Parse PPM format (simple, uncompressed)
          parse_ppm_binary(binary, width, height)

        {:error, reason} ->
          {:error, {:rgb_conversion_failed, reason}}
      end
    rescue
      e ->
        Logger.error("Image conversion error: #{inspect(e)}")
        {:error, {:image_conversion_error, e}}
    end
  end

  # Parse PPM binary format to tensor
  defp parse_ppm_binary(binary, width, height) do
    # PPM format: "P6\n{width} {height}\n255\n{raw_rgb_bytes}"
    # Skip header
    case String.split(binary, "\n", parts: 4) do
      [_format, _dims, _max, rgb_data] ->
        rgb_data
        |> Nx.from_binary(:u8)
        |> Nx.reshape({height, width, 3})
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> then(&{:ok, &1})

      _ ->
        {:error, {:ppm_parse_failed, "Invalid PPM format"}}
    end
  rescue
    _e -> {:error, {:ppm_parse_error, binary}}
  end

  @doc "Normalize image for model input (ImageNet normalization)"
  def normalize(tensor) do
    mean = Nx.tensor([0.485, 0.456, 0.406])
    std = Nx.tensor([0.229, 0.224, 0.225])

    tensor
    |> Nx.subtract(mean)
    |> Nx.divide(std)
  end

  @doc "Resize image to target size"
  def resize(image, width, height) do
    image
    |> Image.resize(width, height)
  end

  @doc "Ensure dimensions are divisible by 8 (for VAE encoding)"
  def pad_to_8_divisible(tensor) do
    [h, w, _c] = Nx.shape(tensor)

    new_h = ceil(h / 8) * 8
    new_w = ceil(w / 8) * 8

    pad_h_total = new_h - h
    pad_w_total = new_w - w

    pad_h_before = div(pad_h_total, 2)
    pad_h_after = pad_h_total - pad_h_before

    pad_w_before = div(pad_w_total, 2)
    pad_w_after = pad_w_total - pad_w_before

    tensor
    |> Nx.pad(0.5, [
      {pad_h_before, pad_h_after},
      {pad_w_before, pad_w_after},
      {0, 0}
    ])
  end

  @doc "Save tensor as image"
  def save_tensor(tensor, path) do
    # Convert from (0,1) range to (0,255) uint8
    binary =
      tensor
      |> Nx.multiply(255.0)
      |> Nx.as_type(:u8)
      |> Nx.to_binary()

    case File.write(path, binary) do
      :ok -> {:ok, path}
      {:error, reason} -> {:error, {:save_failed, reason}}
    end
  end
end
