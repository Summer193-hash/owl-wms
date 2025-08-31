import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
import einops

# Import necessary components from your project structure
from owl_wms.configs import Config
from owl_wms.models import get_model_cls
from owl_wms.sampling import get_sampler_cls
from owl_wms.utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

def load_clean_state_dict(model, checkpoint_path, world_size=1):
    """
    Loads a checkpoint, intelligently handling various common prefixes
    (e.g., 'module.', 'core.', 'ema_model.') to match the model's architecture.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle checkpoints saved with EMA
    if 'ema' in checkpoint:
        checkpoint = checkpoint['ema']
    elif 'model' in checkpoint:
        checkpoint = checkpoint['model']

    # Determine the prefix based on DDP and EMA wrapping
    is_ddp = any(k.startswith('module.') for k in checkpoint.keys())
    is_ema = any(k.startswith('ema_model.') for k in checkpoint.keys())
    
    prefix = ""
    if is_ema:
        prefix += "ema_model."
    if is_ddp:
        prefix += "module."

    # If the model is a wrapper (has a .core attribute), add that to the prefix
    if hasattr(model, 'core'):
         prefix += "core."

    # Strip prefixes if they exist
    if prefix:
        cleaned_state_dict = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    else:
        # Check for 'core.' prefix just in case it wasn't caught
        if any(k.startswith('core.') for k in checkpoint.keys()):
            prefix = "core."
            cleaned_state_dict = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
        else:
            cleaned_state_dict = checkpoint

    target_model = model.core if hasattr(model, 'core') else model
    
    # Load the cleaned state dict
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
    model = model.to(device).bfloat16().eval()
    print("Model ready for inference.")

    # 3. Load the VAE Decoder
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
    
    # 4. Prepare Initial Frame and Action Sequence
    # ============================================
    print("Preparing initial data...")
    data_dir = Path(train_cfg.sample_data_kwargs.root_dir)
    
    # Load the first round from the validation set
    latent_files = sorted(list((data_dir / "latents").glob("*.npy")))
    if not latent_files:
        raise FileNotFoundError(f"No latent files found in {data_dir / 'latents'}")
    
    # Load the initial latent frames to provide context to the model
    initial_latents_full = np.load(latent_files[0])
    # The latent is stored as (C, T, H, W), so we transpose it to (T, C, H, W)
    initial_latents_full = torch.from_numpy(initial_latents_full).permute(1, 0, 2, 3)

    context_window = train_cfg.data_kwargs.window_length
    initial_latents = initial_latents_full[args.seed_frame_index : args.seed_frame_index + context_window]
    initial_latents = initial_latents.unsqueeze(0).to(device).bfloat16() / train_cfg.vae_scale
    
    print(f"Loaded initial latents with shape: {initial_latents.shape}")

    # Prepare the action sequence
    if args.action_sequence:
        action_sequence = torch.tensor(args.action_sequence, device=device).long()
    else:
        # Default action sequence: repeat a "punch" (action_id=8)
        print("No action sequence provided. Using default sequence (repeating punch).")
        action_sequence = torch.tensor([8] * args.num_frames, device=device).long()
        
    # The model needs a full sequence of actions, including those for the initial context frames.
    # We'll just repeat the first action for the context part.
    context_actions = action_sequence[0].repeat(context_window)
    full_action_sequence = torch.cat([context_actions, action_sequence]).unsqueeze(0)
    print(f"Full action sequence shape: {full_action_sequence.shape}")
    
    # 5. Initialize the Sampler
    # ==========================
    print("Initializing sampler...")
    sampler_kwargs = train_cfg.sampler_kwargs
    sampler_kwargs['num_frames'] = args.num_frames # Override with user-defined length
    sampler = get_sampler_cls(train_cfg.sampler_id)(**sampler_kwargs)
    
    # 6. Run Inference
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

    # 7. Save Output
    # ================
    # The output from VAE is [B, C, T, H, W] in range [-1, 1]
    video_out = video_out.squeeze(0) # Remove batch dim
    video_out = einops.rearrange(video_out, 'c t h w -> t h w c')
    video_out = ((video_out + 1) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Create and save the video file
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