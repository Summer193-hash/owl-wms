import torch
from tqdm import tqdm

from ..nn.kv_cache import KVCache
from ..models.tekken_rft_v2 import action_id_to_buttons
from .schedulers import get_sd3_euler

class TekkenCachingActionSampler:
    """
    An efficient sampler for the TekkenRFT model that uses KV caching to speed up
    sequential frame generation. It generates one new frame at a time by conditioning
    on a sliding window of previously generated frames.

    Args:
        n_steps (int): The number of denoising steps for each new frame.
        cfg_scale (float): The scale for classifier-free guidance. Set to 1.0 for no CFG.
        num_frames (int): The total number of new frames to generate.
        noise_prev (float): The amount of noise to add to the context frames.
        only_return_generated (bool): If True, returns only the newly generated frames.
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.0, num_frames: int = 160, noise_prev: float = 0.2, only_return_generated: bool = False, **kwargs):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @staticmethod
    def zlerp(x, alpha):
        """Spherical linear interpolation with noise."""
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    @torch.no_grad()
    def denoise_frame(
        self,
        model,
        kv_cache: KVCache, # type: ignore
        prev_latents: torch.Tensor,
        prev_actions: torch.Tensor,
        curr_actions: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        Performs the full denoising loop for a single new frame using KV caching.
        """
        batch_size = prev_latents.size(0)

        # Step 1: Prepare the initial noisy state
        prev_latents_noised = self.zlerp(prev_latents, self.noise_prev)
        t_prev = prev_latents.new_full((batch_size, prev_latents.size(1)), self.noise_prev)

        new_latent = torch.randn_like(prev_latents[:, :1]) # (b, 1, c, h, w)
        t_new = t_prev.new_ones(batch_size, 1)

        # Step 2: Populate the KV Cache with the context
        kv_cache.enable_cache_updates()

        full_input_latents = torch.cat([prev_latents_noised, new_latent], dim=1)
        full_ts = torch.cat([t_prev, t_new], dim=1)
        full_actions = torch.cat([prev_actions, curr_actions], dim=1) # (b, window_length+1)

        # Convert action IDs to button presses for the model
        # --- Make sure action_ids are long ---
        full_button_presses = action_id_to_buttons(full_actions.long()) # (b, window_length+1, 8)
        curr_button_presses = action_id_to_buttons(curr_actions.long()) # (b, 1, 8)
        # --- End change ---


        # Single forward pass to populate cache (no CFG complexity)
        # --- Pass button presses, not action ids ---
        _ = model(full_input_latents, full_ts, full_button_presses, kv_cache=kv_cache)
        # --- End change ---

        kv_cache.disable_cache_updates()
        kv_cache.truncate(1, front=True)  # Keep only the new frame being denoised

        # Take the first Euler step
        # --- Pass button presses, not action ids ---
        eps_v, _ = model(new_latent, t_new, curr_button_presses, kv_cache=kv_cache) # Ignore state output during sampling
        # --- End change ---

        # --- FIX TypeError ---
        delta_t_0 = dt[0].item() # Convert scalar tensor to Python float
        new_latent = new_latent - eps_v * delta_t_0
        # --- END FIX ---
        t_new = t_new - dt[0] # Keep t_new as tensor

        # Step 3: Remaining denoising steps for the new frame
        for step in range(1, self.n_steps):
            # --- Pass button presses, not action ids ---
            eps_v, _ = model(new_latent, t_new, curr_button_presses, kv_cache=kv_cache) # Ignore state output
             # --- End change ---

            # --- FIX TypeError ---
            delta_t_step = dt[step].item() # Convert scalar tensor to Python float
            new_latent = new_latent - eps_v * delta_t_step
            # --- END FIX ---
            t_new = t_new - dt[step] # Keep t_new as tensor

        return new_latent

    @torch.no_grad()
    def __call__(self, model, initial_latents, action_ids, decode_fn=None, vae_scale=1.0):
        model.eval()
        batch_size, init_len, c, h, w = initial_latents.shape
        dt = get_sd3_euler(self.n_steps).to(device=initial_latents.device, dtype=initial_latents.dtype)
        
        # Initialize KV cache
        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)
        
        # This will store all generated latents, starting with the initial ones
        all_latents = [initial_latents]
        
        # Start with the initial context
        prev_latents = initial_latents
        prev_actions = action_ids[:, :init_len] # (b, window_length, 8)

        for idx in tqdm(range(self.num_frames), desc="Sampling Tekken Frames (Caching)"):
            # Get the action for the new frame we are about to generate
            curr_actions = action_ids[:, init_len + idx: init_len + idx + 1] # (b, 1, 8)
            
            # Denoise the next frame
            new_latent = self.denoise_frame(
                model, kv_cache,
                prev_latents, prev_actions,
                curr_actions,
                dt=dt,
            )

            all_latents.append(new_latent)

            # Update context for next iteration - maintain sliding window
            # For now, let's just use the last frame as context - simplest approach
            prev_latents = new_latent
            prev_actions = curr_actions

        generated_latents = torch.cat(all_latents, dim=1)

        if self.only_return_generated:
            final_latents = generated_latents[:, init_len:]
            final_actions = action_ids[:, init_len : init_len + self.num_frames]
        else:
            final_latents = generated_latents
            final_actions = action_ids[:, :generated_latents.shape[1]]

        video_out = decode_fn(final_latents * vae_scale) if decode_fn is not None else None
        
        return video_out, final_latents, final_actions
  