defmodule SeeThroughBurrito.ModelCacheTest do
  @moduledoc """
  Tests for ModelCache GenServer.
  Verifies model caching and tokenizer loading patterns.
  """

  use ExUnit.Case, async: false

  describe "ModelCache basic operations" do
    test "get_model returns placeholder model" do
      cache_dir = System.tmp_dir!()
      {:ok, _pid} = SeeThroughBurrito.ModelCache.start_link(cache_dir: cache_dir)

      {:ok, model} = SeeThroughBurrito.ModelCache.get_model(
        :clip_l,
        "openai/clip-vit-large-patch14",
        type: :clip_text
      )

      assert is_map(model)
      # Type is converted to string in the cache
      assert model.type == "clip_text"
      assert model.id == "openai/clip-vit-large-patch14"
    end

    test "get_tokenizer returns placeholder tokenizer" do
      cache_dir = System.tmp_dir!()
      {:ok, _pid} = SeeThroughBurrito.ModelCache.start_link(cache_dir: cache_dir)

      {:ok, tokenizer} = SeeThroughBurrito.ModelCache.get_tokenizer(
        :clip_tokenizer,
        "openai/clip-vit-large-patch14"
      )

      assert is_map(tokenizer)
      assert tokenizer.type == :tokenizer
      assert tokenizer.id == "openai/clip-vit-large-patch14"
    end

    test "model cache entries have correct structure" do
      cache_dir = System.tmp_dir!()
      {:ok, _pid} = SeeThroughBurrito.ModelCache.start_link(cache_dir: cache_dir)

      {:ok, model} = SeeThroughBurrito.ModelCache.get_model(
        :structural_test,
        "test/model",
        type: :clip_text
      )

      # Should have these keys
      assert Map.has_key?(model, :id)
      assert Map.has_key?(model, :type)
      assert Map.has_key?(model, :cache_dir)
    end

    test "tokenizer cache entries have correct structure" do
      cache_dir = System.tmp_dir!()
      {:ok, _pid} = SeeThroughBurrito.ModelCache.start_link(cache_dir: cache_dir)

      {:ok, tokenizer} = SeeThroughBurrito.ModelCache.get_tokenizer(
        :tokenizer_struct_test,
        "test/model"
      )

      # Should have these keys
      assert Map.has_key?(tokenizer, :id)
      assert Map.has_key?(tokenizer, :type)
      assert Map.has_key?(tokenizer, :cache_dir)
    end

    test "cache_dir is created if it doesn't exist" do
      temp_dir = Path.join(System.tmp_dir!(), "test_cache_#{System.unique_integer()}")
      {:ok, _pid} = SeeThroughBurrito.ModelCache.start_link(cache_dir: temp_dir)

      assert File.exists?(temp_dir)
    end
  end
end
