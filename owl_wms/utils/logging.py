import torch.distributed as dist
import wandb
import torch

import einops as eo

import numpy as np
from .vis import draw_frames # Assuming .vis contains draw_frames
from .vis_tekken import draw_tekken_frames # Assuming .vis_tekken contains draw_tekken_frames
from moviepy.editor import ImageSequenceClip, CompositeVideoClip
from moviepy.audio.AudioClip import AudioArrayClip

import os
import traceback # Import traceback for detailed error printing

class LogHelper:
    """
    Helps get stats across devices/grad accum steps

    Can log stats then when pop'd will get them across
    all devices (averaged out).
    For gradient accumulation, ensure you divide by accum steps beforehand.
    """
    def __init__(self):
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        self.data = {}

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().item()
        val = data / self.world_size
        if key in self.data:
            self.data[key].append(val)
        else:
            self.data[key] = [val]

    def log_dict(self, d):
        for (k,v) in d.items():
            self.log(k,v)

    def pop(self):
        reduced = {k : sum(v) for k,v in self.data.items()}

        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, reduced)

            final = {}
            for d in gathered:
                for k,v in d.items():
                    if k not in final:
                        final[k] = v
                    else:
                        final[k] += v
        else:
            final = reduced

        self.data = {}
        return final

# --- REPLACED to_wandb FUNCTION ---
@torch.no_grad()
def to_wandb(x, actions, format='mp4', gather=False, max_samples=8, fps=30):
    """
    Creates a WandB Video object with Tekken overlays.
    Expects x: [B, C, T, H, W] tensor on CPU, range [-1, 1]. <--- NOTE: Changed expected input format
    Expects actions: [B, T] tensor on CPU, action IDs.
    """
    print(f"[to_wandb] Received video tensor shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
    print(f"[to_wandb] Received actions tensor shape: {actions.shape}, dtype: {actions.dtype}, device: {actions.device}")

    try:
        # --- Input Validation and Preparation ---
        if x is None or actions is None:
            print("[to_wandb] ERROR: Received None for video or actions.")
            return None

        # Ensure input tensor x is in B, C, T, H, W format
        if x.dim() != 5:
            print(f"[to_wandb] ERROR: Expected video tensor dim 5 (B, C, T, H, W), got {x.dim()}")
            return None
        # Transpose to B, T, C, H, W for processing
        x = x.permute(0, 2, 1, 3, 4) # B, C, T, H, W -> B, T, C, H, W

        x = x[:max_samples].cpu() # Work with max_samples on CPU
        actions = actions[:max_samples].cpu()


        if actions.dim() != 2 or actions.shape[0] != x.shape[0] or actions.shape[1] != x.shape[1]:
             print(f"[to_wandb] ERROR: Actions shape {actions.shape} incompatible with video shape {x.shape}")
             # Attempt to fix if just length mismatch and B=1
             if actions.dim() == 2 and x.dim() == 5 and actions.shape[0] == x.shape[0]:
                 print(f"[to_wandb] Attempting to slice actions to match video length {x.shape[1]}")
                 actions = actions[:, :x.shape[1]]
                 print(f"[to_wandb] New actions shape: {actions.shape}")
                 if actions.shape[1] != x.shape[1]: # Check again after slicing
                     print("[to_wandb] ERROR: Action length still doesn't match video length after slicing.")
                     return None
             else:
                return None


        x = x.clamp(-1, 1)

        # --- Draw Overlays ---
        print("[to_wandb] Calling draw_tekken_frames...")
        # draw_tekken_frames expects tensor [B, T, C, H, W], returns numpy [B, T, 3, H_ext, W] uint8
        drawn_frames_np = draw_tekken_frames(x, actions)
        print(f"[to_wandb] draw_tekken_frames output shape: {drawn_frames_np.shape}, dtype: {drawn_frames_np.dtype}")

        # --- Arrange into Grid for WandB Video ---
        b, t, c_out, h_out, w_out = drawn_frames_np.shape

        # Simple vertical stack if multiple samples
        if b > 1:
            # Stack batches vertically: (T, C, B*H, W)
            video_for_wandb = drawn_frames_np.transpose(1, 2, 0, 3, 4).reshape(t, c_out, b * h_out, w_out)
        else:
            # Single sample: (T, C, H, W)
             video_for_wandb = drawn_frames_np[0].transpose(0, 1, 2, 3) # T, C, H, W

        print(f"[to_wandb] Final video array shape for WandB: {video_for_wandb.shape}, dtype: {video_for_wandb.dtype}")

        # --- Create WandB Video object ---
        print("[to_wandb] Creating wandb.Video object...")
        wandb_video_object = wandb.Video(video_for_wandb, format=format, fps=fps)
        print("[to_wandb] wandb.Video object created successfully.")
        return wandb_video_object # Return the object directly

    except Exception as e:
        print(f"[to_wandb] ERROR creating video: {e}")
        print(traceback.format_exc()) # Print detailed traceback
        return None # Return None on error
# --- END REPLACED FUNCTION ---


def to_wandb_gif(x, actions, max_samples = 4, format='mp4', fps=16):
    x = x.clamp(-1, 1)
    x = (x + 1) * 127.5
    x = x.to(torch.uint8)
    x = x[:max_samples]
    x = eo.rearrange(x, 'b n c h w -> n c h (b w)' )
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    return wandb.Video(x, format=format, fps=fps)

@torch.no_grad()
def to_wandb_av(x, audio, batch_mouse, batch_btn, gather = False, max_samples = 4):
    # x is [b,n,c,h,w]
    # audio is [b,n,2]
    x = x.clamp(-1, 1)
    x = x[:max_samples].cpu().float()

    if False: #x.shape[2] > 3:
        depth = x[:,:,3:4]
        flow = x[:,:,4:7]
        x = x[:,:,:3]

        depth_gif = to_wandb_gif(depth)
        flow_gif = to_wandb_gif(flow)

        feat = True
    else:
        feat = False

    if audio is not None:
        audio = audio[:max_samples].cpu().float().detach().numpy()

    if dist.is_initialized() and gather:
        gathered_x = [None for _ in range(dist.get_world_size())]
        gathered_audio = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_x, x)
        dist.all_gather(gathered_audio, audio)
        x = torch.cat(gathered_x, dim=0)
        if audio is not None:
            audio = torch.cat(gathered_audio, dim=0)

    # Get labels on frames
    x = draw_frames(x, batch_mouse, batch_btn) # -> [b,n,c,h,w] [0,255] uint8 np

    # Convert both to list of [n,h,w,c] and [n,2] numpy arrays
    x = [np.moveaxis(x[i], 1, -1) for i in range(len(x))]
    if audio is not None:
        audio = [audio[i] for i in range(len(audio))]

    os.makedirs("temp_vids", exist_ok = True)
    paths = [f'temp_vids/temp_{i}.mp4' for i in range(len(x))]
    for i, path in enumerate(paths):
        write_video_with_audio(path, x[i], audio[i] if audio is not None else None)

    if feat:
        return [wandb.Video(path, format='mp4') for path in paths], depth_gif, flow_gif
    else:
        return [wandb.Video(path, format='mp4') for path in paths]

def write_video_with_audio(path, vid, audio, fps=60,audio_fps=44100):
    """
    Writes videos with audio to a path at given fps and sample rate

    :param video: [n,h,w,c] [0,255] uint8 np array
    :param audio: [n,2] stereo audio as np array norm to [-1,1]
    """
    # Create video clip from image sequence
    video_clip = ImageSequenceClip(list(vid), fps=fps)

    if audio is not None:
        # Create audio clip from array
        audio_clip = AudioArrayClip(audio, fps=audio_fps)
        # Combine video with audio
        video_clip = video_clip.set_audio(audio_clip)

    # Write to file
    # Use threads=4 and logger=None to potentially reduce console spam from moviepy
    video_clip.write_videofile(
        path,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        threads=4,
        logger=None # Suppress moviepy console output
    )