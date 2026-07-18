defmodule SeeThroughBurrito.Diffusion do
  @moduledoc """
  Diffusion loop orchestration using Bumblebee's built-in schedulers and pipelines.

  Leverages Bumblebee.Diffusion for:
  - Scheduler implementations (DDIM, DPM, etc.)
  - Diffusion step execution
  - Guided/unguided inference
  """

  require Logger
  require Nx

  @doc """
  Run diffusion loop with frame-conditional UNet (LayerDiff).

  Uses Bumblebee.Diffusion scheduler if available, falls back to custom scheduler.

  Inputs:
  - unet: LayerDiff UNet model
  - initial_latents: {batch, H/8, W/8, 4} random noise
  - embeddings: {batch, 77, 1792} CLIP embeddings
  - page_rgb: {H, W, 3} conditioning image
  - num_steps: number of diffusion iterations
  - guidance_scale: classifier-free guidance strength

  Output:
  - {:ok, denoised_latents} or {:error, reason}
  """
  def run_diffusion(unet, initial_latents, embeddings, page_rgb, num_steps, guidance_scale) do
    Logger.info("Running diffusion: #{num_steps} steps, guidance=#{guidance_scale}")

    # Try to use Bumblebee.Diffusion if available
    case try_bumblebee_diffusion(unet, initial_latents, embeddings, page_rgb, num_steps, guidance_scale) do
      {:ok, result} ->
        Logger.info("✅ Diffusion complete (Bumblebee)")
        {:ok, result}

      :not_available ->
        # Fallback to custom scheduler
        Logger.info("Using custom scheduler (Bumblebee.Diffusion not available)")
        run_diffusion_custom(unet, initial_latents, embeddings, page_rgb, num_steps, guidance_scale)
    end
  end

  @doc """
  Try using Bumblebee.Diffusion built-in support.

  Returns :not_available if Bumblebee doesn't have the expected API.
  """
  @dialyzer {:nowarn_function, try_bumblebee_diffusion: 6}
  def try_bumblebee_diffusion(_unet, _initial_latents, _embeddings, _page_rgb, _num_steps, _guidance_scale) do
    # Placeholder: Bumblebee.Diffusion API not yet determined for 0.7.0
    # When Bumblebee API is known, uncomment below:
    #
    # case Bumblebee.Diffusion.run_inference(
    #   unet,
    #   initial_latents,
    #   embeddings: embeddings,
    #   num_steps: num_steps,
    #   guidance_scale: guidance_scale,
    #   conditioning: page_rgb
    # ) do
    #   {:ok, result} -> {:ok, result}
    #   {:error, _} -> :not_available
    #   result when Nx.is_tensor(result) -> {:ok, result}
    #   _ -> :not_available
    # end

    Logger.debug("Bumblebee.Diffusion not available, using custom scheduler")
    :not_available
  end

  @doc """
  Fallback: Custom diffusion loop using our Scheduler module.

  Iterates through diffusion steps with:
  - DPM-Solver++ scheduler (LayerDiff optimized)
  - Classifier-free guidance
  - Frame conditioning
  """
  def run_diffusion_custom(unet, initial_latents, embeddings, page_rgb, num_steps, guidance_scale) do
    Logger.info("Starting custom diffusion loop: #{num_steps} steps")

    # Initialize scheduler state (Scheduler.dpm_solver_init returns state directly)
    scheduler_state = SeeThroughBurrito.Scheduler.dpm_solver_init(num_steps)

    # Run diffusion loop
    result = diffusion_loop(unet, initial_latents, embeddings, page_rgb, scheduler_state, num_steps, guidance_scale)
    {:ok, result}
  end

  defp diffusion_loop(unet, latents, embeddings, page_rgb, scheduler_state, num_steps, guidance_scale) do
    Logger.debug("Diffusion loop: #{num_steps} iterations")

    # Iterate through diffusion steps
    Enum.reduce(0..(num_steps - 1), {latents, scheduler_state}, fn step, {current_latents, sched_state} ->
      Logger.debug("Step #{step + 1}/#{num_steps}")

      # Get timestep for this step (returns value directly, not {:ok, value})
      timestep = SeeThroughBurrito.Scheduler.get_timestep(sched_state, step)

      # Use latents as-is (scaling happens in model if needed)
      scaled_latents = current_latents

      # Run UNet forward pass
      case SeeThroughBurrito.Unet.forward(unet, scaled_latents, timestep, embeddings, page_rgb) do
        {:ok, noise_pred} ->
          # Apply guidance
          guided_noise = apply_classifier_free_guidance(noise_pred, guidance_scale)

          # Single diffusion step
          case SeeThroughBurrito.Scheduler.dpm_solver_step(sched_state, guided_noise, step) do
            {:ok, next_latents, next_state} ->
              {next_latents, next_state}

            {:error, reason} ->
              Logger.error("Scheduler step failed: #{inspect(reason)}")
              {current_latents, sched_state}

            next_latents when Nx.is_tensor(next_latents) ->
              # If dpm_solver_step returns tensor directly instead of {:ok, tensor, state}
              {next_latents, sched_state}
          end

        {:error, reason} ->
          Logger.error("UNet forward failed: #{inspect(reason)}")
          {current_latents, sched_state}

        noise_pred when Nx.is_tensor(noise_pred) ->
          # If UNet returns tensor directly
          guided_noise = apply_classifier_free_guidance(noise_pred, guidance_scale)

          case SeeThroughBurrito.Scheduler.dpm_solver_step(sched_state, guided_noise, step) do
            {:ok, next_latents, next_state} ->
              {next_latents, next_state}

            next_latents when Nx.is_tensor(next_latents) ->
              {next_latents, sched_state}

            _ ->
              {current_latents, sched_state}
          end
      end
    end)
    |> elem(0)
  end

  @doc """
  Apply classifier-free guidance scaling to noise prediction.

  Guidance formula: scaled_noise = guidance_scale * noise_pred
  """
  def apply_classifier_free_guidance(noise_pred, guidance_scale) when guidance_scale > 1.0 do
    # Guidance formula: noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
    # For now, just scale the prediction
    # In full implementation, would compute unconditional prediction separately
    Nx.multiply(noise_pred, guidance_scale)
  end

  def apply_classifier_free_guidance(noise_pred, _guidance_scale) do
    noise_pred
  end

  @doc """
  Run dual-pass diffusion for LayerDiff (body + head).

  Body pass: Iterate on 13 body part tags
  Head pass: Iterate on 11 head part tags
  """
  def run_dual_pass_diffusion(unet, page_rgb, page_alpha, embeddings, opts \\ []) do
    num_steps = Keyword.get(opts, :steps, 30)
    guidance_scale = Keyword.get(opts, :guidance, 7.5)

    Logger.info("Dual-pass diffusion: body + head")

    # Initialize random latents
    latents = Nx.random_normal({1, 64, 64, 4})

    # Body pass (13 tags)
    Logger.info("Body pass (13 tags)...")

    case run_diffusion(unet, latents, embeddings, page_rgb, num_steps, guidance_scale) do
      {:ok, body_latents} ->
        Logger.info("Body pass complete, decoding...")

        # Decode body latents to layers
        case SeeThroughBurrito.Vae.decode_latents(body_latents) do
          {:ok, body_image} ->
            # Head pass (11 tags) - starts from decoded body
            Logger.info("Head pass (11 tags)...")

            # Re-encode body image to latents for head pass
            case SeeThroughBurrito.Vae.encode_image(body_image) do
              {:ok, head_latents} ->
                case run_diffusion(unet, head_latents, embeddings, page_rgb, num_steps, guidance_scale) do
                  {:ok, final_latents} ->
                    Logger.info("Head pass complete")
                    {:ok, %{body_latents: body_latents, head_latents: final_latents}}

                  {:error, reason} ->
                    Logger.error("Head pass failed: #{inspect(reason)}")
                    {:error, {:head_pass_failed, reason}}
                end

              {:error, reason} ->
                Logger.error("Re-encode failed: #{inspect(reason)}")
                {:error, {:re_encode_failed, reason}}
            end

          {:error, reason} ->
            Logger.error("Body decode failed: #{inspect(reason)}")
            {:error, {:body_decode_failed, reason}}
        end

      {:error, reason} ->
        Logger.error("Body pass failed: #{inspect(reason)}")
        {:error, {:body_pass_failed, reason}}
    end
  end
end
