import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import einops
import types

# Import necessary components from your project structure
from owl_wms.configs import Config
from owl_wms.models import get_model_cls
from owl_wms.sampling import get_sampler_cls
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from owl_wms.nn.rope import OrthoRoPE

# ======================================================================
# V V V START: UPDATED CORRECTED FORWARD FUNCTION V V V
# ======================================================================
def corrected_rope_forward(self, x: torch.Tensor, offset: int = 0):
    """
    A corrected forward pass specifically for OrthoRoPE that handles mixed
    precision and the "split-tensor" logic.
    """
    seq_len = x.shape[1]
    
    # Ensure cos and sin caches match the input tensor's dtype and device
    cos = self.cos.to(dtype=x.dtype, device=x.device)
    sin = self.sin.to(dtype=x.dtype, device=x.device)
    
    # Get the correct slice of the embeddings for the current sequence length
    cos_emb = cos[offset : offset + seq_len].unsqueeze(1)
    sin_emb = sin[offset : offset + seq_len].unsqueeze(1)
    
    # Split the input tensor into two halves along the last dimension
    x1, x2 = x.chunk(2, dim=-1)

    # Apply the rotary embeddings using the complex number rotation formula.
    # This correctly handles the half-sized embedding dimensions.
    out1 = x1 * cos_emb - x2 * sin_emb
    out2 = x2 * cos_emb + x1 * sin_emb

    # Concatenate the two halves back together
    x_rope = torch.cat([out1, out2], dim=-1)
    return x_rope
# ======================================================================
# ^ ^ ^ END: UPDATED CORRECTED FORWARD FUNCTION ^ ^ ^
# ======================================================================


def load_clean_state_dict(model, checkpoint_path, world_size=1):
    """
    Loads a checkpoint, intelligently handling various common prefixes
    (e.g., 'module.', 'core.', 'ema_model.') to match the model's architecture.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'ema' in checkpoint:
        checkpoint = checkpoint['ema']
    elif 'model' in checkpoint:
        checkpoint = checkpoint['model']

    is_ddp = any(k.startswith('module.') for k in checkpoint.keys())
    is_ema = any(k.startswith('ema_model.') for k in checkpoint.keys())
    
    prefix = ""
    if is_ema:
        prefix += "ema_model."
    if is_ddp:
        prefix += "module."

    if hasattr(model, 'core'):
         prefix += "core."

    if prefix:
        cleaned_state_dict = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    else:
        if any(k.startswith('core.') for k in checkpoint.keys()):
            prefix = "core."
            cleaned_state_dict = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
        else:
            cleaned_state_dict = checkpoint

    target_model = model.core if hasattr(model, 'core') else model
    
    missing_keys, unexpected_keys = target_model.load_state_dict(cleaned_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
        
    print("✅ Model weights loaded successfully.")


def main(args):
    """
    Main inference function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Configuration
    # ========================
    print(f"Loading configuration from: {args.config}")
    cfg = Config.from_yaml(args.config)
    model_cfg = cfg.model
    train_cfg = cfg.train
    
    # 2. Load the World Model
    # ========================
    print("Initializing model...")
    model = get_model_cls(model_cfg.model_id)(model_cfg)
    load_clean_state_dict(model, args.checkpoint)

    # 3. APPLY MONKEY PATCH
    # ========================
    print("Applying RoPE fix for mixed precision...")
    patched_count = 0
    target_to_patch = model.core if hasattr(model, 'core') else model
    for module in target_to_patch.modules():
        if isinstance(module, OrthoRoPE):
            module.forward = types.MethodType(corrected_rope_forward, module)
            patched_count += 1
    
    if patched_count > 0:
        print(f"✅ Patch applied successfully to {patched_count} RoPE module(s).")
    else:
        print("⚠️ Warning: No RoPE modules were found to patch.")

    model = model.to(device).bfloat16().eval()
    print("Model ready for inference.")

    # 4. Load the VAE Decoder
    # ========================
    print("Loading VAE decoder...")
    decoder = get_decoder_only(
        train_cfg.vae_id,
        train_cfg.vae_cfg_path,
        train_cfg.vae_ckpt_path
    )
    decoder = decoder.to(device).bfloat16().eval()
    decode_fn = make_batched_decode_fn(decoder, train_cfg.vae_batch_size, temporal_vae=True)
    print("VAE decoder ready.")
    
    # 5. Prepare Initial Frame and Action Sequence
    # ============================================
    print("Preparing initial data...")
    data_dir = Path(train_cfg.sample_data_kwargs.root_dir)
        
    try:
        first_round_dir = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('round_')])[0]
    except IndexError:
        raise FileNotFoundError(f"No 'round_...' subdirectories found in {data_dir}")

    latent_files = sorted(list((first_round_dir / "latents").glob("*.npy")))
    if not latent_files:
        raise FileNotFoundError(f"No latent files found in {first_round_dir / 'latents'}")
    
    initial_latents_full = np.load(latent_files[0])
    initial_latents_full = torch.from_numpy(initial_latents_full).permute(1, 0, 2, 3)

    context_window = train_cfg.data_kwargs.window_length
    initial_latents = initial_latents_full[args.seed_frame_index : args.seed_frame_index + context_window]
    initial_latents = initial_latents.unsqueeze(0).to(device).bfloat16() / train_cfg.vae_scale
    
    print(f"Loaded initial latents with shape: {initial_latents.shape}")

    if args.action_sequence:
        action_sequence = torch.tensor(args.action_sequence, device=device).long()
    else:
        print("No action sequence provided. Using default sequence (repeating punch).")
        action_sequence = torch.tensor([8] * args.num_frames, device=device).long()
        
    context_actions = action_sequence[0].repeat(context_window)
    full_action_sequence = torch.cat([context_actions, action_sequence]).unsqueeze(0)
    print(f"Full action sequence shape: {full_action_sequence.shape}")
    
    # 6. Initialize the Sampler
    # ==========================
    print("Initializing sampler...")
    sampler_kwargs = train_cfg.sampler_kwargs
    sampler_kwargs['num_frames'] = args.num_frames
    sampler = get_sampler_cls(train_cfg.sampler_id)(**sampler_kwargs)
    
    # 7. Run Inference
    # =================
    print(f"Running inference to generate {args.num_frames} frames...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        video_out, _, _ = sampler(
            model.core,
            initial_latents,
            full_action_sequence,
            decode_fn=decode_fn,
            vae_scale=train_cfg.vae_scale
        )
    print("Inference complete.")

    # 8. Save Output
    # ================
    video_out = video_out.squeeze(0)
    video_out = einops.rearrange(video_out, 'c t h w -> t h w c')
    video_out = ((video_out + 1) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    clip = ImageSequenceClip(list(video_out), fps=30)
    clip.write_videofile(str(output_path), codec='libx264')

    print(f"✅ Video saved successfully to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for the Tekken World Model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/tekken_action.yml", 
        help="Path to the model configuration YAML file."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output/generated_tekken_gameplay.mp4", 
        help="Path to save the generated video."
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=90, 
        help="Number of new frames to generate."
    )
    parser.add_argument(
        '--action_sequence', 
        type=int, 
        nargs='+', 
        default=None,
        help='A sequence of pre-defined action IDs (integers from 0-255). Example: --action_sequence 8 8 0 0 16 16'
    )
    parser.add_argument(
        '--seed_frame_index',
        type=int,
        default=0,
        help="The starting frame index from the validation set to use as context."
    )
    
    args = parser.parse_args()
    main(args)