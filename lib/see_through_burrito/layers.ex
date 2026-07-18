defmodule SeeThroughBurrito.Layers do
  @moduledoc "Semantic layer decomposition using LayerDiff UNet"
  @behaviour SeeThroughBurrito.LayerAdapter

  require Logger
  import Nx.Defn

  @default_layers [
    "background",
    "character_body",
    "face",
    "eyes",
    "eyebrows",
    "nose",
    "mouth",
    "hair_front",
    "hair_back",
    "hair_sides",
    "hair_bangs",
    "hair_accessories",
    "neck",
    "skin",
    "clothes_upper",
    "clothes_lower",
    "clothes_outerwear",
    "gloves",
    "shoes",
    "socks",
    "accessories_head",
    "accessories_neck",
    "accessories_hand",
    "accessories_waist"
  ]

  @doc "Decompose image into semantic layers"
  @dialyzer {:nowarn_function, decompose: 2}
  def decompose(latents, opts \\ []) do
    layers_to_extract = Keyword.get(opts, :layers, @default_layers)
    steps = Keyword.get(opts, :steps, 30)

    Logger.info("Decomposing into #{length(layers_to_extract)} layers")

    case load_layerdiff_model(opts) do
      {:ok, unet_serving} ->
        decompose_layers(unet_serving, latents, layers_to_extract, steps, opts)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp decompose_layers(unet, latents, layers, steps, opts) do
    # Extract adapters from opts, with defaults
    pipeline_adapter = Keyword.get(opts, :pipeline, SeeThroughBurrito.Pipeline)
    encoder_adapter = Keyword.get(opts, :encoder, SeeThroughBurrito.Encoder)

    results =
      layers
      |> Enum.map(&extract_layer(unet, latents, &1, steps, opts, pipeline_adapter, encoder_adapter))
      |> Enum.filter(&match?({:ok, _}, &1))
      |> Enum.map(fn {:ok, result} -> result end)

    if length(results) == length(layers) do
      {:ok, results}
    else
      {:error, {:layer_extraction_failed, "Some layers failed"}}
    end
  end

  @dialyzer {:nowarn_function, extract_layer: 7}
  defp extract_layer(unet, latents, layer_name, steps, opts, pipeline_adapter, encoder_adapter) do
    Logger.debug("Extracting layer: #{layer_name}")

    # Run diffusion conditioned on layer prompt
    prompt = "#{layer_name} layer"

    case pipeline_adapter.encode_prompt(prompt, nil, nil, opts) do
      {:ok, embeddings} ->
        case pipeline_adapter.run_diffusion(unet, latents, embeddings, Keyword.merge(opts, steps: steps)) do
          {:ok, outputs} ->
            # Decode final latent to image
            final_latent = List.last(outputs)

            case encoder_adapter.decode_from_latents(final_latent, opts) do
              {:ok, layer_image} ->
                {:ok, %{name: layer_name, image: layer_image}}

              {:error, reason} ->
                Logger.error("Failed to decode layer #{layer_name}: #{inspect(reason)}")
                {:error, reason}
            end

          {:error, reason} ->
            Logger.error("Diffusion failed for layer #{layer_name}: #{inspect(reason)}")
            {:error, reason}
        end

      {:error, reason} ->
        Logger.error("Prompt encoding failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc "Extract alpha/transparency information from layers"
  defn extract_alpha(layer_rgb) do
    # Convert RGB to grayscale as base alpha
    weights = Nx.tensor([0.299, 0.587, 0.114])
    alpha = Nx.dot(layer_rgb, weights)
    Nx.clip(alpha, 0.0, 1.0)
  end

  @doc "Composite layers in Z-order using over blending"
  def composite_layers(layers, alphas) when is_list(layers) and is_list(alphas) do
    # Simple over compositing - accumulate weighted contributions
    Enum.zip(layers, alphas)
    |> Enum.reduce(nil, fn {layer, alpha}, acc ->
      alpha_expanded = Nx.new_axis(alpha, -1)
      weighted = Nx.multiply(layer, alpha_expanded)

      case acc do
        nil -> weighted
        result -> Nx.add(result, weighted)
      end
    end)
  end

  defp load_layerdiff_model(opts) do
    model_id = Keyword.get(opts, :layerdiff_model, "shitagaki-lab/layerdiff-unet")
    cache_dir = Keyword.get(opts, :cache_dir, "/tmp/see-through-models")

    Logger.info("Loading LayerDiff model: #{model_id}")

    case SeeThroughBurrito.Models.load_diffusion(model_id, cache_dir: cache_dir) do
      {:ok, model} -> {:ok, model}
      {:error, reason} -> {:error, {:layerdiff_load_failed, reason}}
    end
  end
end
