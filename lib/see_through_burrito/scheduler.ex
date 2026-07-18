defmodule SeeThroughBurrito.Scheduler do
  @moduledoc """
  Diffusion schedulers for noise prediction.
  Ported from see-through-cpp/src/scheduler.cpp

  Implements:
  - DPM-Solver++ 2M SDE (for LayerDiff)
  - DDIM Trailing (for Marigold)
  """

  require Logger

  @doc """
  Initialize DPM-Solver++ 2M SDE scheduler.

  From: see-through-cpp/src/scheduler.cpp:DpmSolverSDE::set_timesteps
  """
  def dpm_solver_init(num_steps) do
    n_train_steps = 1000
    beta_start = 0.00085
    beta_end = 0.012

    # Compute alpha cumulatives (scaled_linear schedule)
    ac = compute_alphas_cumprod(beta_start, beta_end, n_train_steps)

    # "leading" spacing with steps_offset 1
    ratio = n_train_steps / (num_steps + 1)

    timesteps =
      Enum.map(0..(num_steps - 1), fn k ->
        Float.round((num_steps - k) * ratio) + 1
      end)

    # Compute sigmas
    sigmas =
      Enum.map(timesteps, fn t ->
        alpha_t = Enum.at(ac, t)
        :math.sqrt((1.0 - alpha_t) / alpha_t)
      end) ++ [0.0]

    # Return scheduler state
    %{
      type: :dpm_solver_sde,
      timesteps: timesteps,
      sigmas: sigmas,
      step_index: 0,
      lower_order_nums: 0,
      prev_x0: nil
    }
  end

  @doc """
  Initialize DDIM Trailing scheduler (for Marigold).

  From: see-through-cpp/src/scheduler.cpp:DdimTrailing::set_timesteps
  """
  def ddim_trailing_init(num_steps) do
    n_train_steps = 1000
    beta_start = 0.00085
    beta_end = 0.012

    # Compute alpha cumulatives (scaled_linear schedule)
    ac = compute_alphas_cumprod(beta_start, beta_end, n_train_steps)

    # Rescale for zero SNR
    s0 = :math.sqrt(Enum.at(ac, 0))
    st = :math.sqrt(Enum.at(ac, n_train_steps - 1))

    ac_rescaled =
      Enum.with_index(ac, fn alpha, _i ->
        s = (:math.sqrt(alpha) - st) * s0 / (s0 - st)
        s * s
      end)

    final_alpha_cumprod = Enum.at(ac_rescaled, 0)

    # Trailing spacing
    ratio = n_train_steps / num_steps

    timesteps =
      Enum.map(0..(num_steps - 1), fn k ->
        Float.round(n_train_steps - k * ratio) - 1
      end)

    # Return scheduler state
    %{
      type: :ddim_trailing,
      timesteps: timesteps,
      alphas_cumprod: ac_rescaled,
      final_alpha_cumprod: final_alpha_cumprod,
      num_steps: num_steps
    }
  end

  @doc """
  Compute alpha cumulatives from beta schedule (scaled_linear).

  Shared by both schedulers.
  From: see-through-cpp/src/scheduler.cpp:set_timesteps
  """
  def compute_alphas_cumprod(beta_start, beta_end, num_timesteps) do
    # Scaled linear: betas = linspace(sqrt(b0), sqrt(b1), N)^2
    # Then compute cumulative product of (1 - betas)
    Enum.reduce(0..(num_timesteps - 1), {1.0, []}, fn i, {cum_alpha, acc} ->
      # Compute beta_i
      sqrt_beta_start = :math.sqrt(beta_start)
      sqrt_beta_end = :math.sqrt(beta_end)
      sqrt_beta_i = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * i / (num_timesteps - 1)
      beta_i = sqrt_beta_i * sqrt_beta_i

      # Update cumulative alpha: alpha_cumprod_t = alpha_cumprod_{t-1} * (1 - beta_t)
      new_cum_alpha = cum_alpha * (1.0 - beta_i)
      {new_cum_alpha, acc ++ [new_cum_alpha]}
    end)
    |> elem(1)
  end

  @doc """
  DPM-Solver step (for LayerDiff diffusion).

  From: see-through-cpp/src/scheduler.cpp:DpmSolverSDE::step
  """
  def dpm_solver_step(scheduler, sample, eps_pred, noise) do
    # Extract state
    %{
      timesteps: timesteps,
      sigmas: sigmas,
      step_index: step_index,
      lower_order_nums: lower_order_nums,
      prev_x0: prev_x0
    } = scheduler

    n_steps = length(timesteps)

    # Compute alpha_t and sigma_t from sigma
    sigma_t = Enum.at(sigmas, step_index)
    {alpha_t, sigma_st} = sigma_to_alpha_sigma_t(sigma_t)

    # Convert model output (epsilon prediction)
    x0 =
      Nx.subtract(sample, Nx.multiply(eps_pred, sigma_st))
      |> Nx.divide(alpha_t)

    # Check if this is the final step (lower_order_final)
    lower_order_final = step_index == n_steps - 1

    # Get next sigma
    sigma_next = Enum.at(sigmas, step_index + 1)
    {alpha_next, sigma_next_t} = sigma_to_alpha_sigma_t(sigma_next)

    # Compute step coefficients
    lambda_t = :math.log(alpha_next) - :math.log(sigma_next_t)
    lambda_s = :math.log(alpha_t) - :math.log(sigma_st)
    h = lambda_t - lambda_s

    c_sample = sigma_next_t / sigma_st * :math.exp(-h)
    c_x0 = alpha_next * (1.0 - :math.exp(-2.0 * h))
    c_noise = sigma_next_t * :math.sqrt(1.0 - :math.exp(-2.0 * h))

    # Compute next sample
    next_sample =
      if lower_order_nums < 1 or lower_order_final do
        # First-order step
        sample
        |> Nx.multiply(c_sample)
        |> Nx.add(Nx.multiply(x0, c_x0))
        |> Nx.add(Nx.multiply(noise, c_noise))
      else
        # Second-order midpoint step
        x0_prev = prev_x0
        sigma_prev = Enum.at(sigmas, step_index - 1)
        {alpha_prev, sigma_prev_t} = sigma_to_alpha_sigma_t(sigma_prev)

        lambda_prev = :math.log(alpha_prev) - :math.log(sigma_prev_t)
        r0 = (lambda_s - lambda_prev) / h

        derivative =
          Nx.subtract(x0, x0_prev)
          |> Nx.divide(r0)

        sample
        |> Nx.multiply(c_sample)
        |> Nx.add(Nx.multiply(Nx.add(x0, Nx.multiply(derivative, 0.5)), c_x0))
        |> Nx.add(Nx.multiply(noise, c_noise))
      end

    # Update scheduler state
    new_lower_order_nums = min(lower_order_nums + 1, 2)

    scheduler
    |> Map.put(:prev_x0, x0)
    |> Map.put(:lower_order_nums, new_lower_order_nums)
    |> Map.put(:step_index, step_index + 1)
    |> then(&{:ok, next_sample, &1})
  end

  @doc """
  DDIM step (for Marigold depth estimation).

  From: see-through-cpp/src/scheduler.cpp:DdimTrailing::step
  """
  def ddim_step(scheduler, sample, v_pred, step_index) do
    # Extract state
    %{
      timesteps: timesteps,
      alphas_cumprod: alphas_cumprod,
      final_alpha_cumprod: final_alpha_cumprod,
      num_steps: num_steps
    } = scheduler

    # Get timesteps
    t = Enum.at(timesteps, step_index)
    prev_t = t - div(1000, num_steps)

    # Get alpha values
    a_t = Enum.at(alphas_cumprod, t)
    a_prev = if prev_t >= 0, do: Enum.at(alphas_cumprod, prev_t), else: final_alpha_cumprod

    # Compute square roots
    sq_at = :math.sqrt(a_t)
    sq_bt = :math.sqrt(1.0 - a_t)
    sq_ap = :math.sqrt(a_prev)
    sq_bp = :math.sqrt(1.0 - a_prev)

    # v_prediction: x0 = sqrt(a)*x - sqrt(1-a)*v; eps = sqrt(a)*v + sqrt(1-a)*x
    x0 =
      sample
      |> Nx.multiply(sq_at)
      |> Nx.subtract(Nx.multiply(v_pred, sq_bt))

    eps =
      v_pred
      |> Nx.multiply(sq_at)
      |> Nx.add(Nx.multiply(sample, sq_bt))

    # Compute next sample
    next_sample =
      Nx.multiply(x0, sq_ap)
      |> Nx.add(Nx.multiply(eps, sq_bp))

    {:ok, next_sample}
  end

  defp sigma_to_alpha_sigma_t(sigma) do
    alpha_t = 1.0 / :math.sqrt(sigma * sigma + 1.0)
    sigma_t = sigma * alpha_t
    {alpha_t, sigma_t}
  end

  @doc """
  Get the current timestep from scheduler state.
  """
  def get_timestep(scheduler, step_index) do
    Enum.at(scheduler.timesteps, step_index)
  end

  @doc """
  Check if scheduler has completed all steps.
  """
  def is_complete?(scheduler) do
    case scheduler do
      %{type: :dpm_solver_sde, step_index: step_index, timesteps: timesteps} ->
        step_index >= length(timesteps)

      %{type: :ddim_trailing} ->
        true  # DDIM doesn't track completion internally
    end
  end
end
