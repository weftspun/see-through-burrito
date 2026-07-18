defmodule SeeThroughBurrito.BumblebeeAPITest do
  @moduledoc """
  Tests to explore and validate Bumblebee 0.7.0 API signatures.

  This test suite is marked @skip because it requires real model downloads
  from HuggingFace. Run with --include skip to test actual API compatibility.
  """

  use ExUnit.Case, async: false
  require Logger

  @tag :skip
  test "Bumblebee.load_model for CLIP text encoder" do
    Logger.info("Testing CLIP model loading...")

    # Test basic model loading
    assert {:ok, model} = Bumblebee.load_model(
      {:hf, "openai/clip-vit-large-patch14"},
      architecture: :clip_text_model,
      type: :text
    )

    assert is_map(model) or is_struct(model)
    Logger.info("✓ CLIP model loaded successfully")
  end

  @tag :skip
  test "Bumblebee.load_tokenizer for CLIP" do
    Logger.info("Testing CLIP tokenizer loading...")

    {:ok, tokenizer} = Bumblebee.load_tokenizer(
      {:hf, "openai/clip-vit-large-patch14"}
    )

    assert is_map(tokenizer) or is_struct(tokenizer)
    Logger.info("✓ CLIP tokenizer loaded successfully")
  end

  @tag :skip
  test "Bumblebee tokenizer encoding" do
    Logger.info("Testing tokenizer encoding...")

    {:ok, tokenizer} = Bumblebee.load_tokenizer(
      {:hf, "openai/clip-vit-large-patch14"}
    )

    {:ok, token_ids} = Bumblebee.apply_tokenizer(tokenizer, "anime character")

    # Token IDs should be a tensor
    assert Nx.is_tensor(token_ids)
    Logger.info("✓ Tokenization successful: #{inspect(Nx.shape(token_ids))}")
  end

  @tag :skip
  test "Bumblebee VAE model loading" do
    Logger.info("Testing VAE model loading...")

    {:ok, vae} = Bumblebee.load_model(
      {:hf, "stabilityai/sd-vae-ft-mse"},
      architecture: :autoencoder
    )

    assert is_map(vae) or is_struct(vae)
    Logger.info("✓ VAE model loaded successfully")
  end

  @tag :skip
  test "Bumblebee model application (inference)" do
    Logger.info("Testing model inference...")

    # Load model
    {:ok, model} = Bumblebee.load_model(
      {:hf, "openai/clip-vit-large-patch14"},
      architecture: :clip_text_model,
      type: :text
    )

    # Load tokenizer
    {:ok, tokenizer} = Bumblebee.load_tokenizer(
      {:hf, "openai/clip-vit-large-patch14"}
    )

    # Tokenize
    {:ok, token_ids} = Bumblebee.apply_tokenizer(tokenizer, "test")

    # Try inference
    {:ok, output} = Bumblebee.apply_model(model, token_ids)

    # Output should be a tensor or map with tensor
    if Nx.is_tensor(output) do
      Logger.info("✓ Direct tensor output: #{inspect(Nx.shape(output))}")
    else
      Logger.info("✓ Struct output: #{inspect(output)}")
    end
  end

  @tag :skip
  test "Bumblebee model with Axon" do
    Logger.info("Testing Axon integration...")

    {:ok, model} = Bumblebee.load_model(
      {:hf, "openai/clip-vit-large-patch14"},
      architecture: :clip_text_model,
      type: :text
    )

    # Model might have a predict function
    assert respond_to?(model, :predict) or is_struct(model)
    Logger.info("✓ Model ready for Axon inference")
  end

  defp respond_to?(obj, method) when is_atom(method) do
    # Check if object responds to method
    case obj do
      map when is_map(map) -> Map.has_key?(map, method)
      struct when is_struct(struct) -> function_exported?(struct.__struct__, method, 1)
      _ -> false
    end
  end
end
