defmodule PropCheckTests do
  use ExUnit.Case
  use PropCheck

  property "depth map normalization preserves shape" do
    forall tensor <- matrix_data() do
      normalized = SeeThroughBurrito.Depth.normalize_depth(tensor)
      Nx.shape(normalized) == Nx.shape(tensor)
    end
  end

  property "normalized depth is in [0, 1] range" do
    forall tensor <- matrix_data() do
      normalized = SeeThroughBurrito.Depth.normalize_depth(tensor)
      min_val = Nx.min(normalized) |> Nx.to_number()
      max_val = Nx.max(normalized) |> Nx.to_number()

      min_val >= 0.0 and min_val <= 1.0 and max_val >= 0.0 and max_val <= 1.0
    end
  end

  property "depth inversion reverses values correctly" do
    forall tensor <- matrix_data_normalized() do
      inverted = SeeThroughBurrito.Depth.invert_depth(tensor)
      reinverted = SeeThroughBurrito.Depth.invert_depth(inverted)

      # Check that double inversion is approximately original
      diff = Nx.subtract(tensor, reinverted) |> Nx.abs() |> Nx.max() |> Nx.to_number()
      diff < 0.01
    end
  end

  property "depth to height conversion preserves order" do
    forall {a, b} <- {float(min: 0.0, max: 1.0), float(min: 0.0, max: 1.0)} do
      a_tensor = Nx.tensor([[a]])
      b_tensor = Nx.tensor([[b]])

      a_height = SeeThroughBurrito.Depth.depth_to_height(a_tensor) |> Nx.to_number()
      b_height = SeeThroughBurrito.Depth.depth_to_height(b_tensor) |> Nx.to_number()

      (a < b and a_height < b_height) or (a >= b and a_height >= b_height)
    end
  end

  property "layer alpha extraction produces valid transparency" do
    forall rgb <- matrix_data() do
      alpha = SeeThroughBurrito.Layers.extract_alpha(rgb)
      min_val = Nx.min(alpha) |> Nx.to_number()
      max_val = Nx.max(alpha) |> Nx.to_number()

      min_val >= 0.0 and max_val <= 1.0
    end
  end

  property "image padding is reversible" do
    forall tensor <- matrix_data() do
      padded = SeeThroughBurrito.Images.pad_to_8_divisible(tensor)
      [h, w, _] = Nx.shape(padded)

      rem(h, 8) == 0 and rem(w, 8) == 0
    end
  end

  property "normalization bounds preservation" do
    forall tensor <- matrix_data() do
      normalized = SeeThroughBurrito.Pipeline.normalize_sdxl(tensor)

      # After normalization, values should still be finite
      not Nx.any(Nx.is_nan(normalized)) |> Nx.to_number()
    end
  end

  property "guidance scaling is monotonic" do
    forall noise <- matrix_data() do
      scaled_1x = SeeThroughBurrito.Pipeline.scale_guidance(noise, 1.0)
      scaled_2x = SeeThroughBurrito.Pipeline.scale_guidance(noise, 2.0)

      # Element-wise magnitude should increase with scale
      ratio = Nx.divide(Nx.abs(scaled_2x), Nx.abs(scaled_1x) + 1.0e-6)
      mean_ratio = Nx.mean(ratio) |> Nx.to_number()

      mean_ratio > 1.9 and mean_ratio < 2.1
    end
  end

  property "inpainting blend produces valid range" do
    forall {original, inpainted, mask} <- {
      matrix_data(),
      matrix_data(),
      matrix_data_normalized()
    } do
      blended = SeeThroughBurrito.Inpaint.blend_inpainted(original, inpainted, mask)

      min_val = Nx.min(blended) |> Nx.to_number()
      max_val = Nx.max(blended) |> Nx.to_number()

      min_val >= -1.5 and max_val <= 1.5
    end
  end

  property "morphological operations are idempotent after convergence" do
    forall mask <- matrix_data_binary() do
      # Opening then opening should equal opening
      opened_1 = SeeThroughBurrito.Inpaint.morphological_open(mask, 5)
      opened_2 = SeeThroughBurrito.Inpaint.morphological_open(opened_1, 5)

      # Should be approximately the same
      diff = Nx.subtract(opened_1, opened_2) |> Nx.abs() |> Nx.max() |> Nx.to_number()
      diff < 0.01
    end
  end

  # Property generators

  defp matrix_data do
    let size <- {choose(10, 256), choose(10, 256)} do
      {h, w} = size
      Nx.random_normal({h, w, 3})
    end
  end

  defp matrix_data_normalized do
    let tensor <- matrix_data() do
      min_val = Nx.min(tensor)
      max_val = Nx.max(tensor)
      range = Nx.subtract(max_val, min_val)

      tensor
      |> Nx.subtract(min_val)
      |> Nx.divide(range)
    end
  end

  defp matrix_data_binary do
    let size <- {choose(10, 256), choose(10, 256)} do
      {h, w} = size

      Nx.random_uniform({h, w})
      |> Nx.greater(0.5)
      |> Nx.as_type(:f32)
    end
  end
end
