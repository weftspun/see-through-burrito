defmodule SeeThroughBurrito.Clip do
  @moduledoc """
  CLIP text encoder for tag-to-embedding conversion.
  Ported from see-through-cpp/src/clip.cpp

  Provides:
  - Tag encoding to embeddings via dual CLIP models (ViT-L and ViT-G)
  - Text tokenization
  - Batch processing for efficiency
  """

  require Logger
  @behaviour SeeThroughBurrito.ModelAdapter

  @doc """
  Encode a list of tags to embeddings.

  Returns: {:ok, %{embeddings: [...], pooled: [...]}}

  From: see-through-cpp/src/clip.cpp:encode_tags
  """
  @dialyzer {:nowarn_function, encode_tags: 2}
  def encode_tags(tags, opts \\ []) do
    Logger.info("Encoding #{length(tags)} tags via CLIP")

    # Load CLIP models (CLIP-L and CLIP-G for dual embedding)
    clip_l_model_id = Keyword.get(opts, :clip_l_model, "openai/clip-vit-large-patch14")
    clip_g_model_id = Keyword.get(opts, :clip_g_model, "openai/clip-vit-g-14")

    with {:ok, tokenizer_l} <- load_tokenizer(clip_l_model_id, opts),
         {:ok, tokenizer_g} <- load_tokenizer(clip_g_model_id, opts),
         {:ok, model_l} <- load_model(clip_l_model_id, opts),
         {:ok, model_g} <- load_model(clip_g_model_id, opts) do
      # Tokenize all tags
      token_ids_l = Enum.map(tags, &tokenize(&1, tokenizer_l))
      token_ids_g = Enum.map(tags, &tokenize(&1, tokenizer_g))

      # Run inference on both models
      with {:ok, embeddings_l} <- run_inference(model_l, token_ids_l),
           {:ok, embeddings_g} <- run_inference(model_g, token_ids_g) do
        # Concatenate embeddings: [F, 77, 768] + [F, 77, 1024] → [F, 77, 1792]
        embeddings = Nx.concatenate([embeddings_l, embeddings_g], axis: 2)

        # Compute pooled embeddings (mean of tokens)
        pooled = compute_pooled_embeddings(embeddings)

        {:ok, %{embeddings: embeddings, pooled: pooled}}
      else
        {:error, reason} -> {:error, {:inference_failed, reason}}
      end
    else
      {:error, reason} -> {:error, {:model_load_failed, reason}}
    end
  end

  @doc """
  Tokenize a text string using CLIP tokenizer.
  Returns token IDs (padded to 77 tokens, CLIP standard).
  """
  def tokenize(text, tokenizer) do
    Logger.debug("Tokenizing: #{String.slice(text, 0..30)}")

    # Use ModelServing layer for consistent API
    SeeThroughBurrito.ModelServing.encode_text(tokenizer, text)
  end

  @doc """
  Load CLIP tokenizer from HuggingFace or cache.
  """
  def load_tokenizer(model_id, opts) do
    Logger.debug("Loading tokenizer for #{model_id}")

    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    # Use ModelServing layer for consistent API
    SeeThroughBurrito.ModelServing.load_tokenizer(model_id, cache_dir: cache_dir)
  end

  @doc """
  Load CLIP model from HuggingFace or cache via Bumblebee.
  """
  def load_model(model_id, opts) do
    Logger.debug("Loading CLIP model: #{model_id}")

    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    # Use ModelServing layer for consistent Bumblebee API handling
    SeeThroughBurrito.ModelServing.load_model(model_id, :clip_text, cache_dir: cache_dir)
  end

  @doc """
  Run CLIP text encoder inference on token IDs.
  Returns embeddings for the input tokens.
  """
  def run_inference(model, token_ids) do
    Logger.debug("Running CLIP inference on #{length(token_ids)} samples")

    case model do
      nil ->
        {:error, {:invalid_model, "Model is nil"}}

      _model ->
        # Use ModelServing layer for consistent API
        SeeThroughBurrito.ModelServing.run_inference(model, token_ids)
    end
  end

  @doc """
  Compute pooled embeddings (mean over sequence dimension).
  Mean of all 77 token embeddings for each sample.
  """
  def compute_pooled_embeddings(embeddings) do
    # embeddings: {num_samples, 77, embedding_dim}
    # pooled: {num_samples, embedding_dim}
    Nx.mean(embeddings, axes: [1])
  end

end
