from ema_pytorch import EMA
from pathlib import Path
from tqdm import tqdm
import wandb
import gc

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer
from ..utils import freeze, Timer
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..muon import init_muon
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_gif, to_wandb
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

class TekkenRFTTrainerV2(BaseTrainer):
    """
    A specialized trainer for the TekkenRFT model with periodic evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = get_model_cls(self.model_cfg.model_id)(self.model_cfg)

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.total_step_counter = 0

        # VAE decoder is only needed on the main process for logging samples.
        self.decoder = None
        if self.rank == 0:
            self.decoder = get_decoder_only(
                self.train_cfg.vae_id,
                self.train_cfg.vae_cfg_path,
                self.train_cfg.vae_ckpt_path
            )
            freeze(self.decoder)

    def save(self):
        save_dict = {
            'model': self.get_module().state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        if hasattr(self.train_cfg, 'resume_ckpt') and self.train_cfg.resume_ckpt is not None:
            print(f"Loading weights from checkpoint: {self.train_cfg.resume_ckpt}")
            # Load the raw state dict directly
            state_dict = super().load(self.train_cfg.resume_ckpt)

            # --- NEW LOADING LOGIC ---
            try:
                # Try loading directly into the main model (and handle potential DDP/compile prefixes)
                self.get_module().load_state_dict(state_dict, strict=False)
                print("Successfully loaded weights into the main model.")

                # Also load into the EMA model for consistency at the start
                # Need to adjust keys if model was compiled (_orig_mod prefix)
                ema_state_dict = {}
                model_prefix = "_orig_mod." if list(state_dict.keys())[0].startswith("_orig_mod.") else ""
                ema_prefix = "module." if self.world_size > 1 else "" # DDP prefix for EMA

                for k, v in state_dict.items():
                    # Strip potential compile prefix from the source key
                    if k.startswith(model_prefix):
                        key_base = k[len(model_prefix):]
                    else:
                        key_base = k
                    # Add the DDP prefix expected by EMA's underlying model
                    ema_key = f"ema_model.{ema_prefix}{key_base}"
                    ema_state_dict[ema_key] = v

                self.ema.load_state_dict(ema_state_dict, strict=False)
                print("Successfully loaded weights into EMA model.")

                # Since we are loading raw weights, reset step counter
                self.total_step_counter = 0
                print("Reset step counter to 0 as loading raw weights.")

            except Exception as e:
                print(f"Error loading state dict: {e}")
                print("Ensure the checkpoint contains compatible model weights.")
                # Optional: raise e # Re-raise if you want the script to stop on error

            # --- END NEW LOADING LOGIC ---
        else:
            print("No resume_ckpt specified, starting from scratch.")

    def get_sample_data(self):
        """Create a fresh sample batch for evaluation"""
        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.n_samples, **self.train_cfg.sample_data_kwargs)
        return next(iter(sample_loader))

    def eval_step(self, sampler, decode_fn):
        """
        Runs a single evaluation step, generating a video sample and logging it.
        This function will only execute on the rank 0 process.
        """
        # --- Only Rank 0 executes this ---
        if self.rank != 0:
            return {}

        print("\n[Rank 0] ========= STARTING EVAL STEP =========") # Unique message

        # Get fresh sample data each time to avoid StopIteration
        try:
            vid_for_sample, actions_for_sample, _ = self.get_sample_data()
        except Exception as e:
            # --- Print error only on Rank 0 ---
            print(f"[Eval Rank 0] ERROR: Failed to get sample data: {e}")
            return {}

        initial_latents = vid_for_sample.cuda().bfloat16() / self.train_cfg.vae_scale

        # Ensure enough actions for sampling length
        num_required_actions = initial_latents.shape[1] + sampler.num_frames
        if actions_for_sample.shape[1] < num_required_actions:
             num_repeats = (num_required_actions // actions_for_sample.shape[1]) + 1
             actions_for_sample = actions_for_sample.repeat(1, num_repeats, 1) # (b, t*repeat, 8)

        # Get action IDs (adjust slicing if needed based on sampler requirements)
        action_ids = actions_for_sample.cuda()[:, :num_required_actions, -1].int() # (b, t_req, 8) -> (b, t_req)

        # --- Print initial shapes (Rank 0 only) ---
        print(f"[Eval Rank 0] Initial latents shape: {initial_latents.shape}, min: {initial_latents.min():.2f}, max: {initial_latents.max():.2f}")
        print(f"[Eval Rank 0] Action IDs shape (for sampler): {action_ids.shape}")

        sampled_latents = None # Initialize
        video_out = None       # Initialize
        out_actions = None     # Initialize
        wandb_videos = None    # Initialize wandb_videos

        try:
            with torch.no_grad(), torch.amp.autocast('cuda', torch.bfloat16):
                ema_core = self.get_module(ema=True).core # Simplified get_module

                # --- Step 1: Get latents from sampler ---
                print("[Rank 0] Calling sampler...") # Unique message
                _, sampled_latents, out_actions = sampler( # Ignore the first value (video_out is None here)
                ema_core,
                initial_latents,
                action_ids,
                decode_fn=None, # Get raw latents first
                vae_scale=self.train_cfg.vae_scale
                )

                if sampled_latents is None:
                     print("[Rank 0] ERROR: Sampler returned None latents!")
                     # Cleanup before returning
                     del sampled_latents, out_actions, initial_latents, action_ids, vid_for_sample, ema_core
                     gc.collect(); torch.cuda.empty_cache()
                     return {}
                print(f"[Rank 0] Sampler output latents shape: {sampled_latents.shape}, NaN: {torch.isnan(sampled_latents).any()}")

                # --- Step 2: Decode ---
                if decode_fn is not None:
                    print("[Rank 0] Calling decode_fn...") # Unique message
                    video_out = decode_fn(sampled_latents * self.train_cfg.vae_scale)
                    if video_out is None:
                        print("[Rank 0] ERROR: decode_fn returned None!")
                         # Cleanup before returning
                        del sampled_latents, out_actions, initial_latents, action_ids, vid_for_sample, ema_core, video_out
                        gc.collect(); torch.cuda.empty_cache()
                        return {}
                    print(f"[Rank 0] Decoded video shape: {video_out.shape}, NaN: {torch.isnan(video_out).any()}")
                else:
                    print("[Rank 0] ERROR: decode_fn is None, cannot create video.")
                    # Cleanup before returning
                    del sampled_latents, out_actions, initial_latents, action_ids, vid_for_sample, ema_core
                    gc.collect(); torch.cuda.empty_cache()
                    return {}

            # --- Step 3: Create WandB video ---
            if video_out is not None:
                print("[Rank 0] Calling to_wandb...") # Unique message
                # Ensure correct permutation B, C, T, H, W for to_wandb utility
                T_video = video_out.shape[1]
                if out_actions is None: # Use original action_ids if sampler didn't return them
                    out_actions = action_ids[:, initial_latents.shape[1] : initial_latents.shape[1] + sampled_latents.shape[1]] # Slice to match sampled length
                wandb_videos = to_wandb(video_out.cpu(), out_actions.cpu(), format='mp4', fps=30)
                if wandb_videos is None or (isinstance(wandb_videos, list) and len(wandb_videos) > 0 and wandb_videos[0] is None):
                    print("[Rank 0] ERROR: to_wandb returned None or invalid video object!")
                    wandb_videos = None # Ensure we don't try to log invalid data
                else:
                    print("[Rank 0] Successfully created wandb video object.")
            else:
                print("[Rank 0] ERROR: video_out is None after decode step.")

        except Exception as e:
             print(f"[Rank 0] ERROR during eval_step try block: {e}")
             import traceback
             traceback.print_exc()
             wandb_videos = None # Ensure we don't log on error

        finally:
             # Cleanup regardless of errors
             print("[Rank 0] Cleaning up eval_step resources...") # Unique message
             del video_out, sampled_latents, out_actions, initial_latents, actions_for_sample, action_ids, vid_for_sample, ema_core
             gc.collect()
             torch.cuda.empty_cache()

        print("[Rank 0] ========= FINISHED EVAL STEP =========") # Unique message

        # Only return the dictionary if wandb_videos is valid
        if wandb_videos:
            return {"samples": wandb_videos}
        else:
            return {} # Return empty dict if sample creation failed

    def train(self):
        torch.cuda.set_device(self.local_rank)

        self.model = self.model.cuda().train()

        if self.train_cfg.compile:
            print("Compiling the main model...")
            self.model = torch.compile(self.model)

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        decode_fn, sampler = None, None
        if self.rank == 0:
            self.decoder = self.decoder.cuda().eval().bfloat16()

            if self.train_cfg.compile:
                print("Compiling the VAE decoder...")
                self.decoder = torch.compile(self.decoder, mode="reduce-overhead", fullgraph=True)

            decode_fn = make_batched_decode_fn(self.decoder, batch_size=self.train_cfg.vae_batch_size, temporal_vae=True)
            sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        self.ema = EMA(self.model, beta=0.999, update_every=1)
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        accum_steps = max(1, self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size)
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        metrics = LogHelper()
        if self.rank == 0:
            pass
            # wandb.watch(self.get_module(), log='all')

        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        print(f"Data loader created with {len(loader)} batches.")

        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            if self.world_size > 1 and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)

            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.train_cfg.epochs}", disable=self.rank != 0):
                batch_vid, batch_actions, batch_states = [t.cuda() for t in batch]

                # NOTE: This is your current action handling logic. For best results, consider
                # converting the full 8-bit vector to an integer ID.
                # print(f'Shape of batch_vid: {batch_vid.shape}, actions: {batch_actions.shape}, states: {batch_states.shape}')
                action_ids = batch_actions[:, :, -1].int() # (b, t, 8) -> (b, t)

                # --- These lines belong outside the ctx block for clarity ---
                batch_vid_scaled = batch_vid.bfloat16() / self.train_cfg.vae_scale
                batch_sta_scaled = batch_states.bfloat16() # Assuming states are normalized 0-1

                # Average across the 8 action steps (dim 2) to get one target per latent frame
                batch_sta_target = batch_sta_scaled.mean(dim=2) # Shape [B, S, 3]
                # --- End lines moved outside ctx ---

                with ctx:
                    B, S = batch_vid_scaled.shape[:2]

                    # 1. Sample timesteps and noise video (like original RFT)
                    with torch.no_grad():
                        ts = torch.randn(B, S, device=batch_vid_scaled.device, dtype=batch_vid_scaled.dtype).sigmoid()
                        z_video = torch.randn_like(batch_vid_scaled)
                        lerpd_video = batch_vid_scaled * (1 - ts[:, :, None, None, None]) + z_video * ts[:, :, None, None, None]
                        target_video = z_video - batch_vid_scaled # Target is the velocity

                    # 2. Call the model - it now returns two predictions
                    # Make sure the model's forward can accept ts!
                    pred_video, pred_states = self.model(lerpd_video, action_ids=action_ids, ts=ts)

                    # --- CORRECTED RANK CHECK PLACEMENT ---
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"[Explicit Rank Check {dist.get_rank() if dist.is_initialized() else 'N/A'}] Shape pred_states: {pred_states.shape} | Shape batch_sta_scaled: {batch_sta_scaled.shape} | Shape batch_sta_target: {batch_sta_target.shape}") # DEBUG PRINT
                    # --- END CORRECTION ---

                    # 3. Calculate separate losses
                    video_loss = torch.nn.functional.mse_loss(pred_video, target_video)
                    state_loss = torch.nn.functional.mse_loss(pred_states, batch_sta_target) # Compare prediction with ground truth states

                    # 4. Combine losses (adjust weighting factor 0.1 as needed)
                    loss = video_loss + 1.0 * state_loss # Assuming increased weight
                    # --- END MODIFICATIONS ---

                    loss = loss / accum_steps # Apply accumulation scaling AFTER combining

                loss.backward()
                metrics.log('diffusion_loss', video_loss.item()) # Log original video loss as diffusion_loss
                metrics.log('video_loss', video_loss.item())    # Also log video loss separately
                metrics.log('state_loss', state_loss.item())    # Log the new state loss

                local_step += 1

                if (local_step) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.ema.update()

                    self.total_step_counter += 1

                    wandb_dict = metrics.pop()
                    if self.rank == 0:
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']

                    if self.total_step_counter % self.train_cfg.sample_interval == 0:
                        # Pass sampler and decode_fn instead of sample_loader
                        eval_wandb_dict = self.eval_step(sampler, decode_fn)
                        if self.rank == 0:
                            wandb_dict.update(eval_wandb_dict)
                    # Add a barrier here to make all processes wait for rank 0 to finish sampling
                    self.barrier()

                    if self.rank == 0:
                        wandb.log(wandb_dict)

                    if self.total_step_counter % self.train_cfg.save_interval == 0 and self.rank == 0:
                        self.save()

                    self.barrier()


if __name__ == '__main__':
    import sys
    import yaml
    import os
    from pathlib import Path
    from ..configs import Config
    from ..sampling import get_sampler_cls
    from ..data import get_loader
    from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Environment variables loaded from .env file")
    except ImportError:
        # If python-dotenv is not installed, try to load manually
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            print("✓ Environment variables loaded manually from .env file")
        else:
            print("⚠ No .env file found, WANDB functionality may not work")

    # Load config from yaml file
    config_path = Path("configs/tekken_dit_video.yml")
    print(f"Loading config from: {config_path}")

    try:
        # Use the proper Config.from_yaml method
        cfg = Config.from_yaml(config_path)
        print("✓ Config loaded successfully")
        print(f"Model: {cfg.model.model_id}")
        print(f"Sample size: {cfg.model.sample_size}")
        print(f"N frames: {cfg.model.n_frames}")

        # Create trainer instance (rank 0 only for testing)
        trainer = TekkenRFTTrainer(cfg.train, cfg.wandb, cfg.model, global_rank=0, local_rank=0, world_size=1)
        print("✓ Trainer created successfully")

        # Test model initialization
        print(f"✓ Model loaded: {type(trainer.model).__name__}")

        # Test VAE decoder setup
        if trainer.decoder is not None:
            print("✓ VAE decoder loaded successfully")
            decode_fn = make_batched_decode_fn(trainer.decoder, cfg.train.vae_batch_size)
            print("✓ Decode function created")
        else:
            print("✗ VAE decoder failed to load")
            sys.exit(1)

        # Test sampler creation
        try:
            sampler_cls = get_sampler_cls(cfg.train.sampler_id)
            if sampler_cls is not None:
                sampler = sampler_cls(**cfg.train.sampler_kwargs)
                print("✓ Sampler created successfully")
            else:
                print(f"✗ Sampler class not found: {cfg.train.sampler_id}")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Sampler creation failed: {e}")
            sys.exit(1)

        # Test data loader creation
        try:
            # Try to get sample data config, fallback to regular data config
            sample_data_id = getattr(cfg.train, 'sample_data_id', cfg.train.data_id)
            sample_data_kwargs = getattr(cfg.train, 'sample_data_kwargs', cfg.train.data_kwargs)

            sample_loader = get_loader(
                sample_data_id,
                cfg.train.n_samples,
                **sample_data_kwargs
            )
            if sample_loader is not None:
                print("✓ Sample data loader created successfully")
            else:
                print("✗ Data loader creation returned None")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Data loader creation failed: {e}")
            print("This might be due to missing data files in the specified directory")
            data_dir = cfg.train.data_kwargs.get('root_dir', 'not specified') if hasattr(cfg.train, 'data_kwargs') else 'not specified'
            print(f"Data directory: {data_dir}")
            sys.exit(1)

        # Test eval step
        print("\n=== Testing eval_step ===")
        try:
            # Initialize EMA for eval
            trainer.ema = EMA(trainer.model, beta=0.999, update_every=1)

            # Move model to GPU and set to eval mode for testing
            if torch.cuda.is_available():
                trainer.model = trainer.model.cuda().eval()
                trainer.decoder = trainer.decoder.cuda().eval().bfloat16()
                print("✓ Model and decoder moved to GPU")
            else:
                print("⚠ CUDA not available, running on CPU (slower)")
                trainer.model = trainer.model.eval()
                trainer.decoder = trainer.decoder.eval()

            # Test the eval step with new method signature
            eval_results = trainer.eval_step(sampler, decode_fn)

            if eval_results and "samples" in eval_results:
                print("✓ Eval step completed successfully!")
                print(f"✓ Generated samples returned")
            else:
                print("✗ Eval step returned empty results")

        except Exception as e:
            print(f"✗ Eval step failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n=== Test Summary ===")
        print("Basic eval code test completed. Check above for any errors.")

    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        print("Make sure you're running from the correct directory")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)