import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from ..nn.embeddings import TimestepEmbedding, ActionEmbedding
from ..nn.attn import DiT, FinalLayer


class TekkenRFTCoreV2(nn.Module):
    """
    A Rectified Flow Transformer core for Tekken that uses cross-attention between
    action embeddings and video patch tokens, now with an additional state predictor head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_buttons = 8

        assert config.backbone == "dit", "This model requires a DiT backbone."
        self.transformer = DiT(config)

        # Embeddings
        self.t_embed = TimestepEmbedding(config.d_model)
        self.action_embed = ActionEmbedding(config.n_buttons, config.d_model)

        # Projection layers
        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        # ✅ New prediction head (for p1_health, p2_health, timer)
        self.state_predictor = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        # Metadata
        self.config.tokens_per_frame = self.config.sample_size[0] * self.config.sample_size[1]
        self.uncond = config.uncond

    def forward(self, x, t, button_presses, has_controls=None, kv_cache=None):
        """
        Args:
            x: [B, T, C, H, W] input latent tensor
            t: [B, T] timesteps
            button_presses: [B, T, N_buttons]
            has_controls: classifier-free guidance mask
        """
        b, n, c, h, w = x.shape # n is the sequence length (T or S)

        # Time + action conditioning
        t_cond = self.t_embed(t)                           # [B, T, D]
        action_tokens = self.action_embed(button_presses)  # [B, T, 8, D]
        action_emb = action_tokens.mean(dim=2)             # [B, T, D]
        # Note: cond_emb is not directly used by DiT's AdaLN, only t_cond is.
        # It might be used elsewhere if your DiT implementation differs.

        # Flatten latents
        x_tokens = eo.rearrange(x, 'b t c h w -> b t (h w) c')
        x_tokens = self.proj_in(x_tokens)  # [B, T, H*W, D]

        # Prepare transformer input
        b, t, s, d = x_tokens.shape # t is sequence length (T or S), s is tokens per frame (H*W)
        transformer_input = x_tokens.view(b, t * s, d)
        # DiT uses t_cond expanded for AdaLN modulation
        cond_for_transformer = t_cond.unsqueeze(2).expand(b, t, s, d).contiguous().view(b, t * s, d)

        # Transformer forward
        processed_tokens = self.transformer(transformer_input, cond_for_transformer, kv_cache) # Shape [B, T*S, D]

        # --- CORRECTED STATE PREDICTION ---
        # Reshape to separate time and spatial tokens
        processed_reshaped = processed_tokens.view(b, t, s, d) # Shape [B, T, S, D]

        # Get the mean representation across spatial tokens *for each frame* in the sequence
        frame_features = processed_reshaped.mean(dim=2) # Shape [B, T, D]

        # Predict states for *each frame* using the frame features
        predicted_states = self.state_predictor(frame_features) # Shape should now be [B, T, 3]
        # --- END CORRECTION ---

        # Reconstruct latent output using the transformer's output
        # The condition for proj_out should also match the transformer's conditioning (t_cond)
        video_cond_for_proj_out = t_cond.unsqueeze(2).expand(b, t, s, d).contiguous().view(b, t * s, d)
        output_latents = self.proj_out(processed_tokens, video_cond_for_proj_out) # Use processed_tokens directly
        output_video = eo.rearrange(output_latents, 'b (t h w) c -> b t c h w', t=t, h=h, w=w)

        # Return both video latents + state predictions
        return output_video, predicted_states


class TekkenRFTV2(nn.Module):
    """Wrapper for the TekkenRFTV2 that handles rectified flow and classifier-free guidance."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = TekkenRFTCoreV2(config)

    def handle_cfg(self, has_controls=None, cfg_prob=None):
        if cfg_prob is None:
            cfg_prob = self.config.cfg_prob
        if cfg_prob <= 0.0 or has_controls is None:
            return has_controls

        pct_without = 1.0 - has_controls.float().mean()
        if pct_without < cfg_prob:
            needed = cfg_prob - pct_without
            needed_frac = needed / has_controls.float().mean()
            b = has_controls.shape[0]
            mask = (torch.rand(b, device=has_controls.device) <= needed_frac) & has_controls
            has_controls = has_controls & (~mask)
        return has_controls

    def noise(self, tensor, ts):
        """Apply rectified flow noise interpolation."""
        z = torch.randn_like(tensor)
        lerp = tensor * (1 - ts) + z * ts
        target = z - tensor
        return lerp, target

    def forward(self, x, action_ids=None, cfg_prob=None, has_controls=None, ts=None):
        """
        Computes the video prediction and state prediction.
        Note: Loss calculation is moved to the trainer.
        """
        B, S = x.size(0), x.size(1)

        if has_controls is None:
            has_controls = torch.ones(B, device=x.device, dtype=torch.bool)

        if action_ids is None:
            has_controls = torch.zeros_like(has_controls)
            button_presses = torch.zeros(B, S, self.config.n_buttons, device=x.device, dtype=torch.float)
        else:
            # --- Ensure action_ids is long ---
            button_presses = action_id_to_buttons(action_ids.long())  # (B, T, 8)
            # --- End Change ---


        has_controls = self.handle_cfg(has_controls, cfg_prob)

        # --- Ensure ts is passed correctly ---
        if ts is None:
             with torch.no_grad():
                ts = torch.randn(B, S, device=x.device, dtype=x.dtype).sigmoid()

        lerpd_video, target_video = self.noise(x, ts[:, :, None, None, None])
        # --- End Change ---


        # Run the main model
        # --- Pass ts to the core model ---
        pred_video, pred_states = self.core(lerpd_video, ts, button_presses, has_controls)
        # --- End Change ---


        # Return predictions (trainer calculates loss)
        return pred_video, pred_states


def action_id_to_buttons(action_id: torch.Tensor):
    """Convert action IDs to 8-bit button press tensors [B, N, 8]."""
    # --- Ensure action_id is long for bitwise ops ---
    action_id = action_id.long()
    # --- End Change ---
    bit_positions = torch.arange(8, device=action_id.device, dtype=action_id.dtype) # Use action_id's dtype for bit_positions too
    action_expanded = action_id.unsqueeze(-1)
    bit_positions = bit_positions.unsqueeze(0).unsqueeze(0)
    buttons = (action_expanded >> bit_positions) & 1
    # --- Output float for ActionEmbedding ---
    return buttons.float()
    # --- End Change ---