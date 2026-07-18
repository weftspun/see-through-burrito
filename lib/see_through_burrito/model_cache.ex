defmodule SeeThroughBurrito.ModelCache do
  @moduledoc """
  GenServer-based model cache for efficient model reuse.

  Loads models from HuggingFace once and caches them in memory.
  Prevents redundant downloads and initialization.
  """

  use GenServer
  require Logger

  def start_link(opts) do
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    GenServer.start_link(__MODULE__, [cache_dir: cache_dir], name: __MODULE__)
  end

  def init(opts) do
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    File.mkdir_p!(cache_dir)

    {:ok, %{cache_dir: cache_dir, models: %{}, tokenizers: %{}}}
  end

  @doc """
  Get a cached model or load it from HuggingFace.

  Returns: {:ok, model_struct} | {:error, reason}
  """
  def get_model(model_key, model_id, opts \\ []) do
    GenServer.call(__MODULE__, {:get_model, model_key, model_id, opts}, 120_000)
  end

  @doc """
  Get a cached tokenizer or load it from HuggingFace.

  Returns: {:ok, tokenizer_struct} | {:error, reason}
  """
  def get_tokenizer(tokenizer_key, model_id, opts \\ []) do
    GenServer.call(__MODULE__, {:get_tokenizer, tokenizer_key, model_id, opts}, 60_000)
  end

  @doc """
  Clear the cache (for testing).
  """
  def clear_cache do
    GenServer.call(__MODULE__, :clear)
  end

  # Callbacks

  def handle_call({:get_model, model_key, model_id, opts}, _from, state) do
    case Map.get(state.models, model_key) do
      nil ->
        # Model not cached, try to load
        case load_model_from_hf(model_id, state.cache_dir, opts) do
          {:ok, model} ->
            new_state = put_in(state, [:models, model_key], model)
            {:reply, {:ok, model}, new_state}

          {:error, reason} ->
            Logger.error("Failed to load model #{model_key}: #{inspect(reason)}")
            {:reply, {:error, reason}, state}
        end

      model ->
        # Model is cached
        {:reply, {:ok, model}, state}
    end
  end

  def handle_call({:get_tokenizer, tokenizer_key, model_id, opts}, _from, state) do
    case Map.get(state.tokenizers, tokenizer_key) do
      nil ->
        # Tokenizer not cached, try to load
        case load_tokenizer_from_hf(model_id, state.cache_dir, opts) do
          {:ok, tokenizer} ->
            new_state = put_in(state, [:tokenizers, tokenizer_key], tokenizer)
            {:reply, {:ok, tokenizer}, new_state}

          {:error, reason} ->
            Logger.error("Failed to load tokenizer #{tokenizer_key}: #{inspect(reason)}")
            {:reply, {:error, reason}, state}
        end

      tokenizer ->
        # Tokenizer is cached
        {:reply, {:ok, tokenizer}, state}
    end
  end

  def handle_call(:clear, _from, state) do
    {:reply, :ok, %{state | models: %{}, tokenizers: %{}}}
  end

  # Private helpers

  defp load_model_from_hf(model_id, cache_dir, opts) do
    Logger.info("Loading model from HuggingFace: #{model_id}")

    model_type = Keyword.get(opts, :type, :unknown) |> to_string()
    model_cache = Path.join(cache_dir, model_type)
    File.mkdir_p!(model_cache)

    # TODO: Replace with actual Bumblebee.load_model/3 once tested
    # {:ok, model} = Bumblebee.load_model(
    #   {:hf, model_id},
    #   type: model_type,
    #   cache_dir: model_cache,
    #   Keyword.delete(opts, :type)
    # )

    # For now, return placeholder
    {:ok, %{id: model_id, type: model_type, cache_dir: model_cache}}
  end

  defp load_tokenizer_from_hf(model_id, cache_dir, _opts) do
    Logger.info("Loading tokenizer from HuggingFace: #{model_id}")

    tokenizer_cache = Path.join(cache_dir, "tokenizers")
    File.mkdir_p!(tokenizer_cache)

    # TODO: Replace with actual Bumblebee.Tokenizers.load/2 once tested
    # {:ok, tokenizer} = Bumblebee.Tokenizers.load(
    #   {:hf, model_id},
    #   cache_dir: tokenizer_cache
    # )

    # For now, return placeholder
    {:ok, %{id: model_id, type: :tokenizer, cache_dir: tokenizer_cache}}
  end
end
