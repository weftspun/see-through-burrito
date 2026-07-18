defmodule SeeThroughBurrito.TensorOps do
  @moduledoc """
  Core tensor operations for image processing.
  Ported from see-through-cpp/src/image_utils.cpp and ops.cpp
  """

  import Nx.Defn
  require Logger

  @doc """
  Center-square pad and resize: pads to square with pad_value, then resizes to target size.
  Returns {resized_image, scale_factor, pad_info}

  From: see-through-cpp/src/image_utils.cpp:center_square_pad_resize
  """
  def center_square_pad_resize(image_tensor, target_size, pad_value \\ 0.0) do
    # image_tensor: {height, width, channels}
    {h, w, _c} = Nx.shape(image_tensor)

    # Determine square size (max of width and height)
    square_size = max(h, w)

    # Calculate padding needed
    pad_h_total = square_size - h
    pad_w_total = square_size - w

    pad_h_before = div(pad_h_total, 2)
    pad_h_after = pad_h_total - pad_h_before

    pad_w_before = div(pad_w_total, 2)
    pad_w_after = pad_w_total - pad_w_before

    # Pad to square
    padded =
      image_tensor
      |> Nx.pad(pad_value, [
        {pad_h_before, pad_h_after},
        {pad_w_before, pad_w_after},
        {0, 0}
      ])

    # Resize to target
    resized = smart_resize(padded, target_size, target_size)

    # Calculate scale factor: original content → square → target
    scale_factor = square_size / target_size

    pad_info = %{
      square_size: square_size,
      pad_h_before: pad_h_before,
      pad_h_after: pad_h_after,
      pad_w_before: pad_w_before,
      pad_w_after: pad_w_after,
      scale_factor: scale_factor
    }

    {:ok, resized, pad_info}
  rescue
    e ->
      Logger.error("center_square_pad_resize failed: #{inspect(e)}")
      {:error, {:pad_resize_failed, e}}
  end

  @doc """
  Smart resize: uses INTER_AREA when shrinking, INTER_LINEAR when growing.

  From: see-through-cpp/src/image_utils.cpp:smart_resize
  """
  def smart_resize(image_tensor, target_h, target_w) do
    {h, w, _c} = Nx.shape(image_tensor)

    if target_h < h or target_w < w do
      # Shrinking: use area interpolation (averaging)
      resize_area(image_tensor, target_h, target_w)
    else
      # Growing or same: use linear interpolation
      resize_linear(image_tensor, target_h, target_w)
    end
  end

  @doc """
  Resize using area interpolation (for downsampling).
  Simplified: uses average pooling when scaling down.
  """
  def resize_area(image_tensor, target_h, target_w) do
    # For now, use simple scaling
    # Full implementation would match OpenCV's INTER_AREA
    # This is a placeholder using basic interpolation
    {h, w, _c} = Nx.shape(image_tensor)
    _scale_h = h / target_h
    _scale_w = w / target_w

    # Placeholder: just return nearest neighbor for now
    # TODO: Implement proper area interpolation
    image_tensor
  end

  @doc """
  Resize using linear interpolation (for upsampling).
  """
  def resize_linear(image_tensor, target_h, target_w) do
    # Placeholder: just return as-is for now
    # Full implementation would use bilinear interpolation
    # This is a placeholder
    _target_h = target_h
    _target_w = target_w

    # TODO: Implement proper linear interpolation
    # For now, just return the image as-is
    image_tensor
  end

  @doc """
  Normalize image to [0, 1] range and subtract mean for model input.
  Standard preprocessing: (x - mean) / std
  """
  defn normalize_image(image_tensor) do
    # ImageNet normalization
    mean = Nx.tensor([0.485, 0.456, 0.406])
    std = Nx.tensor([0.229, 0.224, 0.225])

    image_tensor
    |> Nx.subtract(mean)
    |> Nx.divide(std)
  end

  @doc """
  Denormalize (reverse normalization) for output visualization.
  """
  defn denormalize_image(tensor) do
    mean = Nx.tensor([0.485, 0.456, 0.406])
    std = Nx.tensor([0.229, 0.224, 0.225])

    tensor
    |> Nx.multiply(std)
    |> Nx.add(mean)
  end

  @doc """
  Clamp tensor values to [min, max] range.
  Used to enforce valid value ranges (e.g., [0, 1] for images).
  """
  defn clamp(tensor, min \\ 0.0, max \\ 1.0) do
    tensor
    |> Nx.max(min)
    |> Nx.min(max)
  end

  @doc """
  Convert RGB to RGBA by adding alpha channel.
  """
  defn rgb_to_rgba(rgb_tensor, alpha_value \\ 1.0) do
    {h, w, _c} = Nx.shape(rgb_tensor)

    # Create alpha channel
    alpha = Nx.broadcast(Nx.tensor(alpha_value), {h, w, 1})

    # Concatenate
    Nx.concatenate([rgb_tensor, alpha], axis: 2)
  end

  @doc """
  Convert RGBA to RGB by dropping alpha channel.
  """
  defn rgba_to_rgb(rgba_tensor) do
    Nx.slice(rgba_tensor, [0, 0, 0], [Nx.axis_size(rgba_tensor, 0), Nx.axis_size(rgba_tensor, 1), 3])
  end

  @doc """
  Alpha blend two images: dst = src * src.alpha + dst * (1 - src.alpha)
  Assumes both are RGBA format.

  From: see-through-cpp/src/see_through.cpp:blend_over
  """
  defn alpha_blend(dst_rgba, src_rgba) do
    # Extract alpha from source
    src_alpha = Nx.slice(src_rgba, [0, 0, 3], [Nx.axis_size(src_rgba, 0), Nx.axis_size(src_rgba, 1), 1])

    # Extract RGB channels
    src_rgb = Nx.slice(src_rgba, [0, 0, 0], [Nx.axis_size(src_rgba, 0), Nx.axis_size(src_rgba, 1), 3])
    dst_rgb = Nx.slice(dst_rgba, [0, 0, 0], [Nx.axis_size(dst_rgba, 0), Nx.axis_size(dst_rgba, 1), 3])

    # Blend RGB: src * src_alpha + dst * (1 - src_alpha)
    blended_rgb =
      Nx.add(
        Nx.multiply(src_rgb, src_alpha),
        Nx.multiply(dst_rgb, Nx.subtract(1.0, src_alpha))
      )

    # Blend alpha
    dst_alpha = Nx.slice(dst_rgba, [0, 0, 3], [Nx.axis_size(dst_rgba, 0), Nx.axis_size(dst_rgba, 1), 1])
    blended_alpha = Nx.add(src_alpha, Nx.multiply(dst_alpha, Nx.subtract(1.0, src_alpha)))

    # Recombine
    Nx.concatenate([blended_rgb, blended_alpha], axis: 2)
  end

  @doc """
  Apply alpha threshold: alpha < threshold → 0
  Removes noise from layer masks.

  From: see-through-cpp/src/see_through.cpp:alpha_floor
  """
  defn alpha_floor(rgba_tensor, threshold \\ 15.0 / 255.0) do
    alpha = Nx.slice(rgba_tensor, [0, 0, 3], [Nx.axis_size(rgba_tensor, 0), Nx.axis_size(rgba_tensor, 1), 1])
    floored_alpha = Nx.select(Nx.less(alpha, threshold), Nx.tensor(0.0), alpha)

    rgb = Nx.slice(rgba_tensor, [0, 0, 0], [Nx.axis_size(rgba_tensor, 0), Nx.axis_size(rgba_tensor, 1), 3])
    Nx.concatenate([rgb, floored_alpha], axis: 2)
  end

  @doc """
  Compute bounding box from alpha channel (non-transparent pixels).
  Returns {x, y, width, height} of the minimal bounding box.

  From: see-through-cpp/src/see_through.cpp:bbox_alpha
  """
  def bbox_alpha(rgba_tensor, threshold \\ 0.01) do
    # Extract alpha channel
    {h, w, _c} = Nx.shape(rgba_tensor)
    alpha = Nx.slice(rgba_tensor, [0, 0, 3], [h, w, 1]) |> Nx.squeeze(axes: [2])

    # Convert to binary mask (threshold > 0)
    mask = Nx.greater(alpha, threshold)

    # Find non-zero positions
    # This is a simplified version - full implementation would compute bounds efficiently
    has_content = Nx.any(mask) |> Nx.to_number()

    if has_content != 0.0 do
      {:ok, %{x: 0, y: 0, width: w, height: h}}  # Placeholder
    else
      {:error, :empty_bbox}
    end
  end

  @doc """
  Tile a single frame to batch for parallel processing.
  Repeats frame N times for batch inference.
  """
  defn tile_to_batch(frame, batch_size) do
    # Frame: {h, w, c}, output: {batch_size, h, w, c}
    Nx.tile(Nx.new_axis(frame, 0), [batch_size, 1, 1, 1])
  end

  @doc """
  Extract first frame from batch (reverse of tile_to_batch).
  """
  defn unbatch_first(batch) do
    # Batch: {batch_size, h, w, c}, output: {h, w, c}
    Nx.slice(batch, [0, 0, 0, 0], [1, Nx.axis_size(batch, 1), Nx.axis_size(batch, 2), Nx.axis_size(batch, 3)])
    |> Nx.squeeze(axes: [0])
  end

  @doc """
  Cast tensor to target dtype if needed.
  Safe casting with type checking.
  """
  def cast_if_needed(tensor, target_type) do
    current_type = Nx.type(tensor)

    if current_type == target_type do
      tensor
    else
      Nx.as_type(tensor, target_type)
    end
  end
end
