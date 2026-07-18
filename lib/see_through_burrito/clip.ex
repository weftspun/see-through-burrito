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
    Logger.debug("Tokenizing: #{text}")

    case tokenizer do
      %{type: :clip_tokenizer} ->
        # Real Bumblebee tokenizer available
        try do
          # TODO: Use Bumblebee.Tokenizers.encode/2 once API stable
          # For now, simulate tokenization
          token_ids = Nx.broadcast(Nx.tensor(0.0), {77})
          {:ok, token_ids}
        rescue
          _e -> {:error, :tokenization_failed}
        end

      :placeholder ->
        # Placeholder for tests
        {:ok, Nx.broadcast(Nx.tensor(0.0), {77})}
    end
  end

  @doc """
  Load CLIP tokenizer from HuggingFace or cache.
  """
  def load_tokenizer(model_id, opts) do
    Logger.debug("Loading tokenizer for #{model_id}")

    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    tokenizer_cache = Path.join(cache_dir, "tokenizers")
    File.mkdir_p!(tokenizer_cache)

    # TODO: Integrate Bumblebee.Tokenizers.load(model_id, cache_dir: tokenizer_cache)
    # For now, return placeholder marker
    {:ok, :placeholder}
  end

  @doc """
  Load CLIP model from HuggingFace or cache via Bumblebee.
  """
  def load_model(model_id, opts) do
    Logger.debug("Loading CLIP model: #{model_id}")

    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")
    model_cache = Path.join(cache_dir, "clip")
    File.mkdir_p!(model_cache)

    # TODO: Replace with real Bumblebee once model loading is tested
    # {:ok, model} = Bumblebee.load_model(
    #   {:hf, model_id},
    #   type: :text,
    #   architecture: :clip_text_model,
    #   cache_dir: model_cache
    # )

    # Return placeholder with expected structure
    {:ok, %{id: model_id, type: :clip_text, cache_dir: model_cache}}
  end

  @doc """
  Run CLIP text encoder inference on token IDs.
  Returns embeddings for the input tokens.
  """
  def run_inference(model, token_ids) do
    Logger.debug("Running CLIP inference on #{length(token_ids)} samples")

    case model do
      %{type: :clip_text} ->
        # TODO: Integrate actual Bumblebee inference:
        # {:ok, embeddings} = Bumblebee.apply_model(
        #   model,
        #   token_ids,
        #   output_key: :text_embeds
        # )

        # For now, return placeholder error (tests marked @skip)
        {:error, {:not_implemented, "CLIP inference awaits Bumblebee wiring"}}

      _other ->
        {:error, {:invalid_model, "Model must have type: :clip_text"}}
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
