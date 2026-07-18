defmodule SeeThroughBurrito.Depth do
  @moduledoc "Monocular depth estimation using Marigold"

  require Logger
  import Nx.Defn

  @doc "Estimate depth map from image"
  def estimate(image_tensor, opts \\ []) do
    resolution = Keyword.get(opts, :depth_resolution, 768)
    steps = Keyword.get(opts, :depth_steps, 4)

    Logger.info("Estimating depth map at #{resolution}px with #{steps} steps")

    case load_marigold_model(opts) do
      {:ok, unet_serving} ->
        case SeeThroughBurrito.Models.run_inference(unet_serving, image_tensor) do
          {:ok, depth_map} ->
            {:ok, normalize_depth(depth_map)}

          {:error, reason} ->
            {:error, {:depth_inference_failed, reason}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Normalize depth map to 0-1 range"
  defn normalize_depth(depth) do
    min_val = Nx.min(depth)
    max_val = Nx.max(depth)
    range = Nx.subtract(max_val, min_val)

    depth
    |> Nx.subtract(min_val)
    |> Nx.divide(range)
    |> Nx.clip(0.0, 1.0)
  end

  @doc "Convert depth to pseudo-height map"
  defn depth_to_height(depth) do
    # Scale depth logarithmically for better visual representation
    log_depth = Nx.log(Nx.add(depth, 1.0))
    normalized = Nx.divide(log_depth, Nx.log(2.0))
    Nx.clip(normalized, 0.0, 1.0)
  end

  @doc "Invert depth (make background far, foreground close)"
  defn invert_depth(depth) do
    Nx.subtract(1.0, depth)
  end

  @doc "Compute surface normals from depth"
  defn compute_normals(depth) do
    # Compute gradients in x and y directions
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

    # Stack as normal vector (x, y, -1)
    Nx.stack([dx, dy, Nx.ones({h, w})], axis: 2)
  end

  defp load_marigold_model(opts) do
    model_id = Keyword.get(opts, :depth_model, "prs-eth/marigold-v1")
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    Logger.info("Loading Marigold depth model: #{model_id}")

    case SeeThroughBurrito.Models.load_diffusion(model_id, cache_dir: cache_dir) do
      {:ok, model} -> {:ok, model}
      {:error, reason} -> {:error, {:marigold_load_failed, reason}}
    end
  end
end
