defmodule SeeThroughBurrito.ModelAdapter do
  @moduledoc "Behavior for model inference adapters (mockable for testing)"

  @callback run_inference(model :: any(), input :: any()) ::
              {:ok, tensor :: Nx.t()} | {:error, reason :: term()}
end

defmodule SeeThroughBurrito.EncoderAdapter do
  @moduledoc "Behavior for VAE encoder/decoder adapters (mockable for testing)"

  @callback encode_to_latents(image :: Nx.t(), opts :: keyword()) ::
              {:ok, latents :: Nx.t()} | {:error, reason :: term()}

  @callback decode_from_latents(latents :: Nx.t(), opts :: keyword()) ::
              {:ok, image :: Nx.t()} | {:error, reason :: term()}
end

defmodule SeeThroughBurrito.LayerAdapter do
  @moduledoc "Behavior for layer decomposition adapters (mockable for testing)"

  @callback decompose(latents :: Nx.t(), opts :: keyword()) ::
              {:ok, layers :: list()} | {:error, reason :: term()}
end

defmodule SeeThroughBurrito.DepthAdapter do
  @moduledoc "Behavior for depth estimation adapters (mockable for testing)"

  @callback estimate(image :: Nx.t(), opts :: keyword()) ::
              {:ok, depth :: Nx.t()} | {:error, reason :: term()}
end

defmodule SeeThroughBurrito.InpaintAdapter do
  @moduledoc "Behavior for inpainting adapters (mockable for testing)"

  @callback fill_holes(layers :: list(), opts :: keyword()) ::
              {:ok, inpainted :: list()} | {:error, reason :: term()}
end

defmodule SeeThroughBurrito.ImageAdapter do
  @moduledoc "Behavior for image I/O adapters (mockable for testing)"

  @callback load(path :: String.t()) ::
              {:ok, image :: any()} | {:error, reason :: term()}

  @callback to_rgb_tensor(image :: any()) ::
              {:ok, tensor :: Nx.t()} | {:error, reason :: term()}
end

defmodule SeeThroughBurrito.PipelineAdapter do
  @moduledoc "Behavior for pipeline orchestration adapters (mockable for testing)"

  @callback preprocess(image :: any(), opts :: keyword()) ::
              {:ok, tensor :: Nx.t()} | {:error, reason :: term()}

  @callback encode_prompt(prompt :: String.t(), tokenizer :: any(), encoder :: any(), opts :: keyword()) ::
              {:ok, embeddings :: Nx.t()} | {:error, reason :: term()}

  @callback run_diffusion(unet :: any(), latents :: Nx.t(), embeddings :: Nx.t(), opts :: keyword()) ::
              {:ok, outputs :: list()} | {:error, reason :: term()}
end
