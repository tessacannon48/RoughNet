import torch

# =============================================================================
# SAMPLING METHODS
# =============================================================================

def p_sample_loop_ddpm(model, scheduler, shape, cond, attrs, device):
    x = torch.randn(shape).to(device)

    for t in reversed(range(scheduler.timesteps)):
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        
        # Model predicts the clean image (x_0)
        pred_x0 = model(x, cond, attrs, t_batch)
        
        # Get the posterior mean and variance from the scheduler
        mean = (scheduler.posterior_mean_coef1[t] * pred_x0 +
                scheduler.posterior_mean_coef2[t] * x)
        
        # Sample from the posterior distribution
        if t > 0:
            noise = torch.randn_like(x)
            var = scheduler.posterior_variance[t]
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

    return x


def p_sample_loop_ddim(model, scheduler, shape, cond, attrs, device, eta=0.0):
    """DDIM sampling method."""
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)

    # DDIM works by skipping timesteps, so we need to define the sampling steps
    timesteps = list(reversed(range(scheduler.timesteps)))
    
    # Loop through the defined sampling steps
    for i, t in enumerate(timesteps):
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        
        # Predict the clean input x_0 
        pred_x0 = model(x, cond, attrs, t_batch)

        # Get the alphas for current and previous timesteps from the scheduler
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        alpha_cumprod_prev_t = scheduler.alpha_cumprod_prev[t]

        # Calculate the deterministic part of the DDIM update
        x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0
        
        # Calculate the direction pointing towards x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev_t - eta**2 * (1 - alpha_cumprod_t / alpha_cumprod_prev_t)) * (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)
        
        # Add stochasticity if eta > 0
        if eta > 0:
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev_t)
            noise = torch.randn_like(x)
        else:
            sigma_t = 0
            noise = torch.zeros_like(x)
            
        # Produce x_t-1 using DDIM formula
        x = x0_part + dir_xt + sigma_t * noise

    return x


def p_sample_loop_plms(model, scheduler, shape, cond, attrs, device, order=4):
    """
    PLMS sampling method.
    Args:
        model: The trained model, assumed to predict the clean image (x_0).
        scheduler: The diffusion scheduler.
        ...
        order: The order of the PLMS method (typically 4).
    """
    # Start from pure random noise x_t
    x = torch.randn(shape).to(device)
    
    # Store previous noise predictions for PLMS
    prev_eps = []

    # Loop through the timesteps from T-1 down to 0
    timesteps = list(reversed(range(scheduler.timesteps)))

    for i, t in enumerate(timesteps):
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

        # Get the previous timestep from the scheduler
        alpha_cumprod_t = scheduler.alpha_cumprod[t]
        alpha_cumprod_prev_t = scheduler.alpha_cumprod_prev[t]

        # Predict the clean image (x_0)
        pred_x0 = model(x, cond, attrs, t_batch)

        # Convert x_0 prediction to noise prediction (epsilon)
        pred_epsilon = (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)
        
        # Add the current epsilon to the history
        prev_eps.append(pred_epsilon)
        
        # Keep only the required number of previous predictions for Adams-Bashforth
        if len(prev_eps) > order:
            prev_eps = prev_eps[-order:]

        # Calculate PLMS-corrected epsilon based on history
        if len(prev_eps) == 1:
            eps = pred_epsilon
        elif len(prev_eps) == 2:
            eps = (3/2) * prev_eps[-1] - (1/2) * prev_eps[-2]
        elif len(prev_eps) == 3:
            eps = (23/12) * prev_eps[-1] - (16/12) * prev_eps[-2] + (5/12) * prev_eps[-3]
        else:
            eps = (55/24) * prev_eps[-1] - (59/24) * prev_eps[-2] + (37/24) * prev_eps[-3] - (9/24) * prev_eps[-4]
        
        # Calculate the next sample using the PLMS update formula
        x0_part = torch.sqrt(alpha_cumprod_prev_t) * pred_x0
        dir_xt  = torch.sqrt(1 - alpha_cumprod_prev_t) * eps

        x = x0_part + dir_xt

    return x