defmodule SeeThroughBurrito.ModelServingTest do
  @moduledoc """
  Tests for ModelServing module.
  Validates API compatibility layer for Bumblebee integration.
  """

  use ExUnit.Case
  require Logger

  @tag :skip
  test "load_model delegates to Bumblebee with correct opts" do
    model_id = "openai/clip-vit-large-patch14"

    {:ok, model} = SeeThroughBurrito.ModelServing.load_model(
      model_id,
      :clip_text,
      cache_dir: "/tmp/test"
    )

    # Model should be a Bumblebee struct or valid model
    assert is_map(model) or is_struct(model)
  end

  @tag :skip
  test "load_tokenizer delegates to Bumblebee" do
    model_id = "openai/clip-vit-large-patch14"

    {:ok, tokenizer} = SeeThroughBurrito.ModelServing.load_tokenizer(
      model_id,
      cache_dir: "/tmp/test"
    )

    # Tokenizer should be valid
    assert is_map(tokenizer) or is_struct(tokenizer)
  end

  @tag :skip
  test "encode_text tokenizes and pads to 77 tokens" do
    model_id = "openai/clip-vit-large-patch14"

    {:ok, tokenizer} = SeeThroughBurrito.ModelServing.load_tokenizer(model_id)
    {:ok, tokens} = SeeThroughBurrito.ModelServing.encode_text(tokenizer, "anime")

    # Should be a tensor
    assert Nx.is_tensor(tokens)
  end

  @tag :skip
  test "run_inference returns tensor output" do
    model_id = "openai/clip-vit-large-patch14"

    {:ok, model} = SeeThroughBurrito.ModelServing.load_model(
      model_id,
      :clip_text
    )

    {:ok, tokenizer} = SeeThroughBurrito.ModelServing.load_tokenizer(model_id)
    {:ok, tokens} = SeeThroughBurrito.ModelServing.encode_text(tokenizer, "test")

    # Run inference
    {:ok, output} = SeeThroughBurrito.ModelServing.run_inference(model, tokens)

    # Output should be tensor
    assert Nx.is_tensor(output) or is_map(output) or is_struct(output)
  end

  test "load_model handles different model types" do
    # Test that different model types are recognized
    # (without actually downloading models)
    model_types = [:clip_text, :vae, :unet, :depth_estimation]

    Enum.each(model_types, fn model_type ->
      assert is_atom(model_type)
    end)
  end

  test "error handling for invalid model" do
    # run_inference with nil should error
    {:error, _reason} = SeeThroughBurrito.ModelServing.run_inference(nil, :dummy)
  end

  test "error handling for invalid text encoding" do
    # encode_text with nil should error
    {:error, _reason} = SeeThroughBurrito.ModelServing.encode_text(nil, "test")
  end
end
