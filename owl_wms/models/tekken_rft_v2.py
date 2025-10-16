import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from ..nn.embeddings import TimestepEmbedding, ActionEmbedding
from ..nn.attn import DiT, FinalLayer

class TekkenRFTCoreV2(nn.Module):
    """
    A Rectified Flow Transformer core for Tekken that uses cross-attention between
    action embeddings and video patch tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_buttons = 8

        assert config.backbone == "dit", "This model requires a DiT backbone."
        self.transformer = DiT(config)
        self.t_embed = TimestepEmbedding(config.d_model)
        self.action_embed = ActionEmbedding(config.n_buttons, config.d_model)
        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        # --- THIS IS THE NEW NEURAL NETWORK ---
        # This small MLP is the 'state_predictor' head.
        self.state_predictor = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 3) # Output: p1_health, p2_health, timer
        )
        # --- END OF NEW NETWORK ---

        self.config.tokens_per_frame = self.config.sample_size[0] * self.config.sample_size[1]
        self.uncond = config.uncond

    def forward(self, x, t, button_presses, has_controls=None, kv_cache=None):
        b, n, c, h, w = x.shape
        t_cond = self.t_embed(t)
        action_tokens = self.action_embed(button_presses)
        action_emb = action_tokens.mean(dim=2)
        cond_emb = t_cond + action_emb

        x_tokens = eo.rearrange(x, 'b t c h w -> b t (h w) c')
        x_tokens = self.proj_in(x_tokens)

        b, t, s, d = x_tokens.shape
        transformer_input = x_tokens.view(b, t * s, d)
        cond = t_cond.unsqueeze(2).expand(b, t, s, d).contiguous().view(b, t * s, d)

        processed_tokens = self.transformer(transformer_input, cond, kv_cache)

        # --- USE THE NEW NETWORK ---
        # We take the transformer's output and feed it to our new state_predictor.
        last_frame_features = processed_tokens.view(b, t, s, d)[:, -1, :, :].mean(dim=1)
        predicted_states = self.state_predictor(last_frame_features)
        # --- END OF USAGE ---

        processed_video_tokens = processed_tokens.view(b, t * s, d)
        video_cond = t_cond.unsqueeze(2).expand(b, t, s, d).contiguous().view(b, t * s, d)
        output_latents = self.proj_out(processed_video_tokens, video_cond)
        
        output = eo.rearrange(output_latents, 'b (t h w) c -> b t c h w', t=t, h=h, w=w)
        
        # --- RETURN BOTH PREDICTIONS ---
        return output, predicted_states
    

class TekkenRFTV2(nn.Module):
    """Wrapper for the TekkenRFTV2 that handles rectified flow."""
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
        z = torch.randn_like(tensor)
        lerp = tensor * (1 - ts) + z * ts
        target = z - tensor
        return lerp, target

    def forward(self, x, action_ids=None, ts=None, cfg_prob=None, has_controls=None):
        """
        This forward pass now just runs the core model and returns its predictions.
        """
        B, S = x.size(0), x.size(1)
        
        if has_controls is None:
            has_controls = torch.ones(B, device=x.device, dtype=torch.bool)
        if action_ids is None:
            has_controls = torch.zeros_like(has_controls)
            button_presses = torch.zeros(B, S, self.config.n_buttons, device=x.device, dtype=torch.float)
        else:
            button_presses = action_id_to_buttons(action_ids)

        has_controls = self.handle_cfg(has_controls, cfg_prob)
        
        if ts is None:
            raise ValueError("The 'ts' tensor must be provided by the trainer.")

        # This now calls the updated core and gets two outputs.
        pred_video, pred_states = self.core(x, ts, button_presses, has_controls)
        
        # It then passes those two outputs on to the trainer.
        return pred_video, pred_states