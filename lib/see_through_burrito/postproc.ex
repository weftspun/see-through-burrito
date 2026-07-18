defmodule SeeThroughBurrito.Postproc do
  @moduledoc """
  Post-processing for decomposed layers.
  Ported from see-through-cpp/src/postproc.cpp

  Provides:
  - Layer thresholding and alpha filtering
  - Bounding box computation and cropping
  - Depth-based layer ordering
  - Layer compositing and alpha blending
  """

  require Logger

  @doc """
  Apply post-processing to decomposed layers.

  Steps:
  1. Alpha thresholding (remove noise)
  2. Bounding box cropping
  3. Depth-based ordering
  4. Final compositing

  From: see-through-cpp/src/postproc.cpp:postprocess_layers
  """
  def postprocess(layers, depth_map, opts \\ []) do
    Logger.info("Post-processing #{length(layers)} layers")

    alpha_threshold = Keyword.get(opts, :alpha_threshold, 15.0 / 255.0)
    bbox_threshold = Keyword.get(opts, :bbox_threshold, 0.01)
    do_crop = Keyword.get(opts, :crop, true)
    do_order = Keyword.get(opts, :depth_order, true)

    layers
    |> threshold_alpha(alpha_threshold)
    |> maybe_crop_to_bbox(do_crop, bbox_threshold)
    |> maybe_order_by_depth(do_order, depth_map)
  end

  @doc """
  Apply alpha thresholding to all layers.

  Values below threshold are set to 0.
  From: see-through-cpp/src/see_through.cpp:alpha_floor
  """
  def threshold_alpha(layers, threshold) when is_list(layers) do
    Logger.debug("Thresholding alpha at #{threshold}")

    Enum.map(layers, fn layer ->
      apply_alpha_threshold(layer, threshold)
    end)
  end

  defp apply_alpha_threshold(rgba_tensor, threshold) do
    # Extract alpha channel
    {h, w, _c} = Nx.shape(rgba_tensor)
    alpha = Nx.slice(rgba_tensor, [0, 0, 3], [h, w, 1]) |> Nx.squeeze(axes: [2])

    # Threshold: alpha < threshold → 0
    floored_alpha = Nx.select(Nx.less(alpha, threshold), Nx.tensor(0.0), alpha)

    # Recombine RGB + thresholded alpha
    rgb = Nx.slice(rgba_tensor, [0, 0, 0], [h, w, 3])
    Nx.concatenate([rgb, Nx.new_axis(floored_alpha, 2)], axis: 2)
  end

  @doc """
  Crop layers to their bounding boxes.

  Each layer is cropped to the minimal rectangle containing non-transparent pixels.
  From: see-through-cpp/src/see_through.cpp:bbox_alpha
  """
  def maybe_crop_to_bbox(layers, false, _threshold), do: layers
  def maybe_crop_to_bbox(layers, true, threshold) do
    Logger.debug("Cropping layers to bounding boxes")

    Enum.map(layers, fn layer ->
      crop_to_bbox(layer, threshold)
    end)
  end

  defp crop_to_bbox(rgba_tensor, threshold) do
    {h, w, _c} = Nx.shape(rgba_tensor)

    # Extract alpha channel
    alpha = Nx.slice(rgba_tensor, [0, 0, 3], [h, w, 1]) |> Nx.squeeze(axes: [2])

    # Binary mask: alpha > threshold
    mask = Nx.greater(alpha, threshold)

    # Find bounding box (simplified: return full tensor if any non-zero)
    # Full implementation would compute actual bounds
    case Nx.any(mask) |> Nx.to_number() do
      0 -> rgba_tensor  # Empty layer, keep as-is
      _ -> rgba_tensor  # TODO: Compute and crop to actual bounds
    end
  end

  @doc """
  Order layers by depth for correct compositing.
  Back-to-front ordering ensures correct visual stacking.
  """
  def maybe_order_by_depth(layers, false, _depth_map), do: layers
  def maybe_order_by_depth(layers, true, depth_map) do
    Logger.debug("Ordering layers by depth")
    SeeThroughBurrito.Marigold.sort_by_depth(layers, depth_map)
  end

  @doc """
  Composite layers with depth-aware blending.

  Renders layers back-to-front using alpha-over blending.
  From: see-through-cpp/src/see_through.cpp:blend_over
  """
  def composite_layers(layers) when is_list(layers) do
    Logger.debug("Compositing #{length(layers)} layers")

    case layers do
      [] -> nil
      [single] -> single
      _ -> Enum.reduce(layers, hd(layers), &alpha_blend/2)
    end
  end

  defp alpha_blend(src, dst) do
    # Use TensorOps for blending
    SeeThroughBurrito.TensorOps.alpha_blend(dst, src)
  end

  @doc """
  Extract layer statistics (bounds, alpha range, etc).

  Useful for validation and optimization.
  """
  def layer_stats(rgba_tensor) do
    {h, w, _c} = Nx.shape(rgba_tensor)

    # Extract alpha
    alpha = Nx.slice(rgba_tensor, [0, 0, 3], [h, w, 1]) |> Nx.squeeze(axes: [2])

    %{
      shape: {h, w},
      alpha_min: Nx.reduce_min(alpha) |> Nx.to_number(),
      alpha_max: Nx.reduce_max(alpha) |> Nx.to_number(),
      alpha_mean: Nx.mean(alpha) |> Nx.to_number(),
      coverage: coverage_percent(alpha)
    }
  end

  defp coverage_percent(alpha) do
    # Percentage of non-zero alpha pixels
    mask = Nx.greater(alpha, 0.0)
    count = Nx.sum(mask) |> Nx.to_number()
    total = Nx.size(alpha)
    Float.round(count / total * 100.0, 2)
  end

  @doc """
  Filter layers by quality metrics.

  Removes very small or transparent layers.
  """
  def filter_by_quality(layers, opts \\ []) do
    Logger.debug("Filtering layers by quality")

    min_coverage = Keyword.get(opts, :min_coverage, 0.1)  # 0.1% minimum
    min_alpha = Keyword.get(opts, :min_alpha, 15.0 / 255.0)

    Enum.filter(layers, fn layer ->
      stats = layer_stats(layer)
      stats.coverage > min_coverage and stats.alpha_max > min_alpha
    end)
  end

  @doc """
  Compute layer depth order from depth map.

  Returns list of {layer_index, average_depth} sorted by depth.
  """
  def compute_layer_order(layers, depth_map) when is_list(layers) do
    Logger.debug("Computing layer depth order")

    layers
    |> Enum.with_index()
    |> Enum.map(fn {layer, idx} ->
      avg_depth = compute_layer_depth(layer, depth_map)
      {idx, avg_depth}
    end)
    |> Enum.sort_by(fn {_idx, depth} -> depth end, :desc)
  end

  defp compute_layer_depth(layer_rgba, depth_map) do
    {h, w, _c} = Nx.shape(layer_rgba)

    # Extract alpha channel
    alpha = Nx.slice(layer_rgba, [0, 0, 3], [h, w, 1]) |> Nx.squeeze(axes: [2])

    # Ensure depth map same shape
    {dh, dw} = Nx.shape(depth_map)

    if {h, w} != {dh, dw} do
      Logger.warning("Layer shape #{h}x#{w} doesn't match depth #{dh}x#{dw}")
      0.5  # Default mid-depth
    else
      # Average depth weighted by alpha
      weighted_depth = Nx.multiply(depth_map, alpha)
      sum_depth = Nx.sum(weighted_depth) |> Nx.to_number()
      sum_alpha = Nx.sum(alpha) |> Nx.to_number()

      if sum_alpha > 0 do
        sum_depth / sum_alpha
      else
        0.5
      end
    end
  end

  @doc """
  Apply morphological opening (erosion → dilation).

  Removes small noise and small holes.
  From: see-through-burrito/lib/inpaint.ex:morphological_open
  """
  def morphological_open(mask, kernel_size \\ 5) do
    Logger.debug("Applying morphological opening (kernel #{kernel_size})")

    # Erode then dilate
    mask
    |> SeeThroughBurrito.Inpaint.erode(kernel_size)
    |> SeeThroughBurrito.Inpaint.dilate(kernel_size)
  end
end
