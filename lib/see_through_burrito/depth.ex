defmodule SeeThroughBurrito.Depth do
  @moduledoc "Monocular depth estimation using Marigold"
  @behaviour SeeThroughBurrito.DepthAdapter

  require Logger

  @doc "Estimate depth map from image"
  def estimate(image_tensor, opts \\ []) do
    _resolution = Keyword.get(opts, :depth_resolution, 768)
    _steps = Keyword.get(opts, :depth_steps, 4)

    Logger.info("Depth estimation requested (awaiting Marigold model integration)")

    case load_marigold_model(opts) do
      {:ok, model} ->
        Logger.info("✓ Marigold model loaded: #{model.id}")
        # TODO: Run inference once Bumblebee Axon integration complete
        {:ok, Nx.broadcast(0.5, Nx.shape(image_tensor))}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Normalize depth map to 0-1 range (outside defn for compatibility)"
  def normalize_depth(depth) do
    min_val = Nx.reduce_min(depth) |> Nx.to_number()
    max_val = Nx.reduce_max(depth) |> Nx.to_number()
    range = max_val - min_val

    if range < 1.0e-6 do
      Nx.broadcast(0.5, Nx.shape(depth))
    else
      depth
      |> Nx.subtract(min_val)
      |> Nx.divide(range)
      |> Nx.clip(0.0, 1.0)
    end
  end

  @doc "Convert depth to pseudo-height map"
  def depth_to_height(depth) do
    depth
    |> Nx.add(1.0)
    |> Nx.log()
    |> Nx.divide(Nx.log(2.0))
    |> Nx.clip(0.0, 1.0)
  end

  @doc "Invert depth (background far, foreground close)"
  def invert_depth(depth) do
    Nx.subtract(1.0, depth)
  end

  @doc "Compute surface normals from depth via Sobel gradients"
  def compute_normals(depth) do
    [h, w] = Nx.shape(depth)

    # Pad depth for gradient computation
    padded = Nx.pad(depth, 0.0, [{1, 1}, {1, 1}])

    # Compute x gradient (Sobel-like)
    dx_left = Nx.slice(padded, [1, 0], [h, w])
    dx_right = Nx.slice(padded, [1, 2], [h, w])
    dx = Nx.subtract(dx_right, dx_left)

    # Compute y gradient
    dy_top = Nx.slice(padded, [0, 1], [h, w])
    dy_bottom = Nx.slice(padded, [2, 1], [h, w])
    dy = Nx.subtract(dy_bottom, dy_top)

    # Stack as normal vector (x, y, 1)
    ones = Nx.broadcast(1.0, {h, w})
    Nx.stack([dx, dy, ones], axis: 2)
  end

  defp load_marigold_model(opts) do
    model_id = Keyword.get(opts, :depth_model, "marigold-unet")
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    Logger.info("Loading Marigold depth model: #{model_id}")

    case SeeThroughBurrito.Models.load_model(model_id, cache_dir: cache_dir) do
      {:ok, model} -> {:ok, model}
      {:error, reason} -> {:error, {:marigold_load_failed, reason}}
    end
  end
end
