defmodule SeeThroughBurrito.Inpaint do
  @moduledoc "Inpainting module for hole filling using LaMa or similar"
  @behaviour SeeThroughBurrito.InpaintAdapter

  require Logger

  @doc "Fill holes in decomposed layers"
  @dialyzer {:nowarn_function, fill_holes: 2}
  def fill_holes(layers, opts \\ []) do
    Logger.info("Inpainting #{length(layers)} layers to fill holes")

    layers
    |> Enum.map(&inpaint_layer(&1, opts))
    |> Enum.filter(&match?({:ok, _}, &1))
    |> Enum.map(fn {:ok, result} -> result end)
    |> case do
      results when length(results) == length(layers) ->
        {:ok, results}

      results ->
        Logger.warning("Only #{length(results)}/#{length(layers)} layers inpainted successfully")
        {:error, {:inpainting_incomplete, results}}
    end
  end

  @dialyzer {:nowarn_function, inpaint_layer: 2}
  defp inpaint_layer(%{name: name, image: image}, opts) do
    Logger.debug("Inpainting layer: #{name}")

    with {:ok, mask} <- detect_holes(image),
         {:ok, inpainted} <- run_lama_inpainting(image, mask, opts) do
      {:ok, %{name: name, image: inpainted}}
    else
      {:error, reason} ->
        Logger.error("Inpainting failed for #{name}: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc "Detect holes/transparent regions in layer"
  def detect_holes(image_rgba) do
    # Assume alpha channel is last dimension [H, W, 4]
    # Extract alpha channel
    [h, w, _c] = Nx.shape(image_rgba)
    alpha = Nx.slice(image_rgba, [0, 0, 3], [h, w, 1]) |> Nx.squeeze(axes: [2])
    # Holes are regions with low alpha
    Nx.less(alpha, 0.5)
  end

  @doc "Apply LaMa inpainting model"
  @dialyzer {:nowarn_function, run_lama_inpainting: 3}
  def run_lama_inpainting(image, mask, opts \\ []) do
    Logger.debug("Running LaMa inpainting")

    case load_lama_model(opts) do
      {:ok, lama_serving} ->
        # Prepare input: concatenate image and mask
        input = Nx.concatenate([image, Nx.new_axis(mask, -1)], axis: 2)

        case SeeThroughBurrito.Models.run_inference(lama_serving, input) do
          {:ok, inpainted} ->
            {:ok, inpainted}

          {:error, reason} ->
            {:error, {:lama_inference_failed, reason}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Morphological opening: erosion followed by dilation"
  def morphological_open(mask, kernel_size \\ 5) do
    eroded = erode(mask, kernel_size)
    dilate(eroded, kernel_size)
  end

  @doc "Morphological dilation: maximum filter over kernel"
  def dilate(mask, kernel_size) do
    dilate_impl(mask, kernel_size)
  end

  @doc "Morphological erosion: minimum filter over kernel"
  def erode(mask, kernel_size) do
    erode_impl(mask, kernel_size)
  end

  # Proper dilation via max pooling (mathematical definition)
  defp dilate_impl(mask, kernel_size) do
    # For each pixel, dilation sets it to the maximum value in its kernel neighborhood
    k = div(kernel_size, 2)

    # Pad with 0 (so missing pixels don't increase values)
    padded = Nx.pad(mask, 0.0, [{k, k}, {k, k}])

    # Use Nx.conv to apply max pooling
    # This is mathematically the proper morphological dilation
    apply_max_pool(padded, kernel_size)
  end

  # Proper erosion via min pooling (mathematical definition)
  defp erode_impl(mask, kernel_size) do
    # For each pixel, erosion sets it to the minimum value in its kernel neighborhood
    k = div(kernel_size, 2)

    # Pad with 1.0 (so missing pixels don't decrease values - treat as foreground)
    padded = Nx.pad(mask, 1.0, [{k, k}, {k, k}])

    # Use custom min pooling
    apply_min_pool(padded, kernel_size)
  end

  # Morphological dilation using window reduce (proper mathematical definition)
  defp apply_max_pool(padded, kernel_size) do
    Nx.window_reduce(padded, 0.0, {kernel_size, kernel_size}, &Nx.max/2)
  end

  # Morphological erosion using window reduce (proper mathematical definition)
  defp apply_min_pool(padded, kernel_size) do
    Nx.window_reduce(padded, 1.0, {kernel_size, kernel_size}, &Nx.min/2)
  end

  @doc "Blend inpainted regions smoothly"
  def blend_inpainted(original, inpainted, mask) do
    # Use mask as blend factor (0 = original, 1 = inpainted)
    mask_expanded = mask |> Nx.new_axis(-1)

    Nx.add(
      Nx.multiply(original, Nx.subtract(1.0, mask_expanded)),
      Nx.multiply(inpainted, mask_expanded)
    )
  end

  defp load_lama_model(opts) do
    model_id = Keyword.get(opts, :lama_model, "facebook/lama")
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    Logger.info("Loading LaMa inpainting model: #{model_id}")

    case SeeThroughBurrito.Models.load_diffusion(model_id, cache_dir: cache_dir) do
      {:ok, model} -> {:ok, model}
      {:error, reason} -> {:error, {:lama_load_failed, reason}}
    end
  end
end
