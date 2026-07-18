ExUnit.start()

defmodule SeeThroughBurrito.TestHelper do
  @moduledoc "Test utilities for GPU-aware testing"

  @doc "Check if CUDA is available"
  def cuda_available? do
    try do
      # Attempt to compile a simple defn function with EXLA
      {:ok, _} = Nx.Defn.Compiler.compile(:test, fn -> Nx.tensor([1]) end, [])
      true
    rescue
      _ -> false
    catch
      _ -> false
    end
  end

  @doc "Skip test if CUDA not available"
  def skip_unless_cuda(_context) do
    if cuda_available?() do
      :ok
    else
      {:skip, "CUDA/ExLA not available in test environment"}
    end
  end

  @doc "Create mock model for testing"
  def mock_model(model_id) do
    %{
      id: model_id,
      path: "/tmp/mock_#{model_id}.safetensors",
      type: :safetensors,
      mock: true
    }
  end

  @doc "Create random tensor for testing"
  def random_tensor(shape, opts \\ []) do
    Nx.random_uniform(shape)
  end
end

# Setup test environment
File.mkdir_p!("/tmp/see-through-test-data")
