defmodule PropCheckTest do
  use ExUnit.Case

  doctest SeeThroughBurrito

  # Property-based tests for numerical correctness
  # These require CUDA/ExLA to be available and compiled
  # To run on GPU: mix test --include gpu
  #
  # Once CUDA is available, uncomment the tests below
  # and remove the :skip tag

  @tag :skip
  test "placeholder: property tests disabled without CUDA" do
    # Property tests require GPU execution
    # Run with: CUDA_AVAILABLE=1 mix test --include skip
    assert true
  end

  # TODO: Uncomment when CUDA/ExLA is available
  #
  # test "depth map normalization preserves shape" do
  #   for _ <- 1..10 do
  #     shape = {32, 32, 3}
  #     tensor = Nx.fill(Nx.tensor(0.5), shape)
  #     normalized = SeeThroughBurrito.Depth.normalize_depth(tensor)
  #     assert Nx.shape(normalized) == Nx.shape(tensor)
  #   end
  # end
  #
  # test "normalized depth is in [0, 1] range" do
  #   tensor = Nx.fill(Nx.tensor(0.7), {16, 16})
  #   normalized = SeeThroughBurrito.Depth.normalize_depth(tensor)
  #   min_val = Nx.reduce_min(normalized) |> Nx.to_number()
  #   max_val = Nx.reduce_max(normalized) |> Nx.to_number()
  #   assert min_val >= 0.0 and max_val <= 1.0
  # end
end
