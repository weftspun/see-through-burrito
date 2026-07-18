defmodule SeeThroughBurrito.Marigold do
  @moduledoc """
  Marigold monocular depth estimation.
  Ported from see-through-cpp (integrated into pipeline)

  Provides:
  - Monocular depth map estimation from RGB images
  - Depth normalization and inversion
  - Surface normal computation
  """

  require Logger
  @behaviour SeeThroughBurrito.DepthAdapter

  @doc """
  Estimate depth from a single image.

  Input: RGB image tensor {height, width, 3}
  Output: depth map {height, width} normalized to [0, 1]

  Marigold configuration (from see-through-cpp):
  - v_prediction (velocity prediction)
  - rescale_betas_zero_snr (zero-SNR rescaling)
  - trailing spacing
  - set_alpha_to_one: false
  - clip_sample: false
  - eta: 0 (deterministic)
  """
  @dialyzer {:nowarn_function, estimate: 2}
  def estimate(image_tensor, opts \\ []) do
    Logger.info("Estimating depth via Marigold")

    model_id = Keyword.get(opts, :model_id, "prs-eth/marigold-v1")
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    steps = Keyword.get(opts, :depth_steps, 4)

    with {:ok, model} <- load_model(model_id, cache_dir),
         {:ok, depth} <- run_depth_inference(model, image_tensor, steps) do
      # Normalize depth to [0, 1]
      normalized = normalize_depth(depth)
      {:ok, normalized}
    else
      {:error, reason} ->
        Logger.error("Depth estimation failed: #{inspect(reason)}")
        {:error, {:depth_estimation_failed, reason}}
    end
  end

  @doc """
  Load Marigold depth model from HuggingFace or cache.
  """
  def load_model(model_id, cache_dir) do
    Logger.debug("Loading Marigold model: #{model_id}")

    # Use ModelServing layer for consistent Bumblebee API handling
    SeeThroughBurrito.ModelServing.load_model(model_id, :depth_estimation, cache_dir: cache_dir)
  end

  @doc """
  Run depth inference via Marigold.
  """
  def run_depth_inference(model, image_tensor, _steps) do
    Logger.debug("Running Marigold depth inference")

    case model do
      nil ->
        {:error, {:invalid_model, "Model is nil"}}

      _model ->
        # Use ModelServing layer for inference
        SeeThroughBurrito.ModelServing.run_inference(model, image_tensor)
    end
  end

  @doc """
  Normalize depth map to [0, 1] range.

  Input: raw depth values (typically unbounded)
  Output: normalized depth in [0, 1]
  """
  def normalize_depth(depth_tensor) do
    Logger.debug("Normalizing depth map")

    # Min-max normalization
    min_depth = Nx.reduce_min(depth_tensor)
    max_depth = Nx.reduce_max(depth_tensor)

    # Avoid division by zero
    range = Nx.subtract(max_depth, min_depth)

    Nx.select(
      Nx.equal(range, 0.0),
      Nx.broadcast(0.5, Nx.shape(depth_tensor)),
      Nx.divide(Nx.subtract(depth_tensor, min_depth), range)
    )
  end

  @doc """
  Invert depth map (swap foreground/background).

  Used for layer ordering: inverted_depth = 1 - depth
  """
  def invert_depth(depth_tensor) do
    Logger.debug("Inverting depth map")
    Nx.subtract(1.0, depth_tensor)
  end

  @doc """
  Compute surface normals from depth map using Sobel gradients.

  Input: depth map {height, width}
  Output: normal map {height, width, 3} with normalized XYZ vectors

  From: see-through-cpp/src/depth.cpp (implied)
  """
  def compute_normals(depth_tensor) do
    Logger.debug("Computing surface normals from depth")

    # TODO: Implement Sobel gradient computation
    # Sobel-X: [-1, 0, 1] kernel
    # Sobel-Y: [-1, 0, 1]^T kernel
    # Normal = normalize(cross(dX, dY))

    # Placeholder: return zero normals until implemented
    {h, w} = Nx.shape(depth_tensor)
    Nx.broadcast(Nx.tensor(0.0), {h, w, 3})
  end

  @doc """
  Get depth statistics for a map.

  Returns: %{min, max, mean, std}
  """
  def depth_stats(depth_tensor) do
    %{
      min: Nx.reduce_min(depth_tensor) |> Nx.to_number(),
      max: Nx.reduce_max(depth_tensor) |> Nx.to_number(),
      mean: Nx.mean(depth_tensor) |> Nx.to_number(),
      std: Nx.standard_deviation(depth_tensor) |> Nx.to_number()
    }
  end

  @doc """
  Sort layers by depth (for correct compositing order).

  Input: layers list, depth map
  Output: layers sorted by average depth (back to front)
  """
  def sort_by_depth(layers, depth_map) when is_list(layers) do
    Logger.debug("Sorting layers by depth")

    # Compute average depth for each layer's alpha channel
    layers_with_depth =
      Enum.map(layers, fn layer ->
        avg_depth = compute_layer_depth(layer, depth_map)
        {layer, avg_depth}
      end)

    # Sort by depth (descending - background first)
    layers_with_depth
    |> Enum.sort_by(fn {_layer, depth} -> depth end, :desc)
    |> Enum.map(fn {layer, _depth} -> layer end)
  end

  defp compute_layer_depth(layer_rgba, depth_map) do
    # Extract alpha channel
    {h, w, _c} = Nx.shape(layer_rgba)
    alpha = Nx.slice(layer_rgba, [0, 0, 3], [h, w, 1]) |> Nx.squeeze(axes: [2])

    # Weight depth by alpha (only consider non-transparent regions)
    weighted_depth = Nx.multiply(depth_map, alpha)
    sum_depth = Nx.sum(weighted_depth)
    sum_alpha = Nx.sum(alpha)

    # Average: sum(depth * alpha) / sum(alpha)
    sum_depth |> Nx.divide(sum_alpha) |> Nx.to_number()
  end

  # DepthAdapter behavior implementation
  def run_inference(_model, _input) do
    {:error, {:not_implemented, "Marigold inference awaits Bumblebee integration"}}
  end
end
