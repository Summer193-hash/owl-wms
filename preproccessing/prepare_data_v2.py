import os
import glob
from tqdm import tqdm
import shutil
import torch
import numpy as np
import time
from encoder import load_vae, load_encoder
import subprocess
import multiprocessing as mp
from functools import partial


def save_video_ffmpeg(video_np, output_path, fps=30):
    """
    Save video frames using ffmpeg.

    Args:
        video_np: numpy array of shape [t, h, w, c] with values 0-255
        output_path: path to save the video
        fps: frames per second
    """
    num_frames, height, width, channels = video_np.shape

    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',  # The input comes from a pipe
        '-an',  # No audio
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as process:
            process.stdin.write(video_np.tobytes())
            process.stdin.close()

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0:
                print(f"Video saved successfully to {output_path}")
            else:
                stderr_output = process.stderr.read().decode('utf-8')
                print(f"FFmpeg failed with return code {return_code}")
                print(f"FFmpeg stderr: {stderr_output}")

    except Exception as e:
        print(f"Error running ffmpeg: {e}")


def test_vae(vae, data_path, enable_tiling=False, encode_separately=False, device='cuda'):
    start_time = time.time()

    print("Loading frames...")
    load_start = time.time()
    frames = get_frames_from_npz(data_path)
    frames = frames.to(device)
    load_time = time.time() - load_start
    print(f"Frame loading time: {load_time:.2f}s")

    length = frames.shape[0]
    # input should have n * 4 + 1 frames
    n_frames_to_keep = length - (length - 1) % 4
    frames = frames[:n_frames_to_keep]
    frames = frames.to(torch.bfloat16)
    frames = frames.permute(1, 0, 2, 3)  # [t, c, h, w] -> [c, t, h, w]
    frames = frames.unsqueeze(0)  # Add batch dimension
    print(f"Input shape: {frames.shape}")  # [1, c, t, h, w]

    frames = (frames / 255.0) * 2 - 1

    # Enable VAE tiling if requested
    if enable_tiling:
        print("Enabling VAE tiling...")
        vae.enable_tiling()
    else:
        print("VAE tiling disabled")

    print("Running VAE inference...")
    inference_start = time.time()
    vae.eval()
    with torch.no_grad():
        if encode_separately:
            encoding_start = time.time()
            clean_latent_dist = vae.encode(frames).latent_dist
            encoding_time = time.time() - encoding_start
            print(f"Encoding time: {encoding_time:.2f}s")

            z0 = clean_latent_dist.sample()

            # 2. Define a small noise level 't'
            # The paper trained the decoder in the [0, 0.2] range. 0.05 is a good starting point.
            t_val = 0.0
            t = torch.tensor([t_val] * z0.shape[0], device=device, dtype=z0.dtype)

            # 3. Create noise and generate the noisy latent, zt
            noise = torch.randn_like(z0)
            zt = (1 - t_val) * z0 + t_val * noise

            # 4. Decode from the *noisy* latent, providing the timestep 't'
            decoding_start = time.time()
            decoded_output = vae.decode(zt, temb=t)
            decoded = decoded_output.sample
            decoding_time = time.time() - decoding_start
            print(f"Decoding time: {decoding_time:.2f}s")
        else:
            # Directly encode and decode in one step
            decoded = vae(frames).sample
            encoding_time = decoding_time = 0.0

    # Disable tiling after inference
    if enable_tiling:
        vae.disable_tiling()

    inference_time = time.time() - inference_start
    print(f"VAE inference time: {inference_time:.2f}s")

    print("Saving videos...")
    save_start = time.time()

    # Save reconstructed video
    reconstructed_tensor = decoded.squeeze(0).permute(1, 2, 3, 0).cpu()
    reconstructed_np = ((reconstructed_tensor + 1) / 2.0 * 255.0).to(torch.uint8).numpy()
    reconstructed_output_path = "reconstructed_video.mp4"
    save_video_ffmpeg(reconstructed_np, reconstructed_output_path, fps=24)

    save_time = time.time() - save_start
    print(f"Video saving time: {save_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n=== Timing Summary ===")
    print(f"Frame loading: {load_time:.2f}s")
    print(f"VAE inference: {inference_time:.2f}s")
    print(f"  - Encoding: {encoding_time:.2f}s")
    print(f"  - Decoding: {decoding_time:.2f}s")
    print(f"Video saving: {save_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tiling enabled: {enable_tiling}")
    print(f"Reconstructed video saved to: {reconstructed_output_path}")


def process_single_file(args):
    """Process a single .npz file on a specific GPU."""
    npz_path, split_output_dir, vae_ckpt_dir, enable_tiling, gpu_id, batch_size = args
    
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    # Clear cache at the start
    torch.cuda.empty_cache()
    
    try:
        # Load VAE on the specific GPU
        vae = load_vae(vae_ckpt_dir, device=device)
        if enable_tiling:
            vae.enable_tiling()
        
        round_name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"GPU {gpu_id}: Processing {round_name}...")

        # Create output directories for the current round
        round_output_dir = os.path.join(split_output_dir, round_name)
        latents_dir = os.path.join(round_output_dir, "latents")
        actions_dir = os.path.join(round_output_dir, "actions")
        states_dir = os.path.join(round_output_dir, "states")

        os.makedirs(latents_dir, exist_ok=True)
        os.makedirs(actions_dir, exist_ok=True)
        os.makedirs(states_dir, exist_ok=True)

        # Load data from .npz file
        data = np.load(npz_path)
        mask = data['valid_frames']
        end_idx = int(np.where(mask == 1)[0][-1])

        images = torch.from_numpy(data['images'][:end_idx]).float()
        video_len = images.shape[0]
        n_frames_to_keep = video_len - (video_len - 1) % 4
        images = images[:n_frames_to_keep]
        actions = data['actions_p1'][:end_idx]  # Using actions_p1, ignoring actions_p2
        states = data['states'][:end_idx]
        data.close()

        print(f"GPU {gpu_id}: Video length: {video_len}, processing {n_frames_to_keep} frames")

        # Auto-determine batch size if not set or if sequence is too long
        effective_batch_size = batch_size
        if batch_size <= 0 or n_frames_to_keep > 200:  # Force batching for long sequences
            effective_batch_size = 64  # Conservative default
            print(f"GPU {gpu_id}: Using automatic batch size of {effective_batch_size}")

        # Process in batches for better memory management
        all_latents = []
        num_batches = (images.shape[0] + effective_batch_size - 1) // effective_batch_size
        
        for i in range(num_batches):
            start_idx = i * effective_batch_size
            end_idx = min((i + 1) * effective_batch_size, images.shape[0])
            batch_images = images[start_idx:end_idx]
            
            try:
                with torch.no_grad():
                    batch_images = batch_images.to(device)
                    batch_images = (batch_images / 255.0) * 2 - 1
                    batch_images = batch_images.to(vae.dtype)
                    batch_images = batch_images.permute(1, 0, 2, 3)  # [t, c, h, w] -> [c, t, h, w]
                    batch_images = batch_images.unsqueeze(0)  # Add batch dimension
                    
                    latent_dist = vae.encode(batch_images).latent_dist
                    latent_sample = latent_dist.sample()
                    batch_latents = latent_sample.to(torch.float16).cpu()
                    all_latents.append(batch_latents.squeeze(0).numpy())
                    
                    # Clear intermediate tensors
                    del batch_images, latent_dist, latent_sample, batch_latents
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print(f"GPU {gpu_id}: OOM on batch {i+1}/{num_batches}, trying smaller batch size")
                torch.cuda.empty_cache()
                
                # Retry with smaller batch size
                smaller_batch_size = max(1, effective_batch_size // 2)
                batch_images = images[start_idx:end_idx]
                
                # Process this batch in even smaller chunks
                sub_batches = (batch_images.shape[0] + smaller_batch_size - 1) // smaller_batch_size
                batch_results = []
                
                for j in range(sub_batches):
                    sub_start = j * smaller_batch_size
                    sub_end = min((j + 1) * smaller_batch_size, batch_images.shape[0])
                    sub_batch = batch_images[sub_start:sub_end]
                    
                    with torch.no_grad():
                        sub_batch = sub_batch.to(device)
                        sub_batch = (sub_batch / 255.0) * 2 - 1
                        sub_batch = sub_batch.to(vae.dtype)
                        sub_batch = sub_batch.permute(1, 0, 2, 3)
                        sub_batch = sub_batch.unsqueeze(0)
                        
                        latent_dist = vae.encode(sub_batch).latent_dist
                        latent_sample = latent_dist.sample()
                        sub_result = latent_sample.to(torch.float16).cpu()
                        batch_results.append(sub_result.squeeze(0).numpy())
                        
                        # Clear memory immediately
                        del sub_batch, latent_dist, latent_sample, sub_result
                        torch.cuda.empty_cache()
                
                # Combine sub-batch results
                combined_result = np.concatenate(batch_results, axis=1)
                all_latents.append(combined_result)
        
        # Combine all batches
        final_latents = np.concatenate(all_latents, axis=1)  # Concatenate along time dimension

        # Save the processed data
        np.save(os.path.join(latents_dir, f"{round_name}_latents.npy"), final_latents)
        np.save(os.path.join(actions_dir, f"{round_name}_actions.npy"), actions)
        np.save(os.path.join(states_dir, f"{round_name}_states.npy"), states)

        if enable_tiling:
            vae.disable_tiling()
        
        print(f"GPU {gpu_id}: Finished processing {round_name}")
        return round_name
        
    except Exception as e:
        print(f"GPU {gpu_id}: Error processing {round_name}: {str(e)}")
        return f"ERROR: {round_name}"
        
    finally:
        # Always clear cache when done
        torch.cuda.empty_cache()


def process_data(data_dir, output_dir, vae_ckpt_dir, enable_tiling=False, num_gpus=2, batch_size=0):
    """
    Processes raw Tekken round data from .npz files using multiple GPUs.
    
    Args:
        data_dir: Directory containing 'train' and 'val' subdirectories
        output_dir: Output directory for processed data
        vae_ckpt_dir: Path to VAE checkpoint
        enable_tiling: Whether to enable VAE tiling
        num_gpus: Number of GPUs to use (default: 2)
        batch_size: Batch size for processing frames (0 = process all at once)
    """
    print("--- Starting Multi-GPU Data Processing ---")
    print(f"Using {num_gpus} GPUs with batch size: {batch_size if batch_size > 0 else 'full sequence'}")

    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"Warning: Only {available_gpus} GPUs available, requested {num_gpus}")
        num_gpus = available_gpus

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    original_start_method = mp.get_start_method(allow_none=True)
    if original_start_method != 'spawn':
        mp.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn' for CUDA compatibility")

    try:
        # Process both train and val directories
        for split in ['train', 'val']:
            split_data_dir = os.path.join(data_dir, split)
            split_output_dir = os.path.join(output_dir, split)
            
            if not os.path.exists(split_data_dir):
                print(f"Directory {split_data_dir} does not exist. Skipping {split} split.")
                continue
                
            # Find all .npz files in the split directory
            npz_files = sorted(glob.glob(os.path.join(split_data_dir, "*.npz")))
            if not npz_files:
                print(f"No .npz files found in {split_data_dir}. Skipping {split} split.")
                continue

            print(f"\nProcessing {split} split: Found {len(npz_files)} rounds to process.")

            # Prepare arguments for multiprocessing
            file_args = []
            for i, npz_path in enumerate(npz_files):
                gpu_id = i % num_gpus  # Distribute files across GPUs
                file_args.append((npz_path, split_output_dir, vae_ckpt_dir, enable_tiling, gpu_id, batch_size))

            # Process files using multiprocessing with spawn method
            with mp.Pool(processes=num_gpus) as pool:
                results = list(tqdm(
                    pool.imap(process_single_file, file_args),
                    total=len(file_args),
                    desc=f"Processing {split} files"
                ))

            print(f"Completed processing {len(results)} files for {split} split")

    finally:
        # Restore original start method if it was changed
        if original_start_method and original_start_method != 'spawn':
            mp.set_start_method(original_start_method, force=True)

    print("\n--- Multi-GPU Data Processing Complete ---")


def create_split(data_dir1, data_dir2, output_dir, split=0.8):
    import random
    
    train_dir = output_dir + "/train"
    val_dir = output_dir + "/val"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Collect files from each directory separately to ensure balanced distribution
    p1_files = sorted(glob.glob(os.path.join(data_dir1, "*.npz")))
    p2_files = sorted(glob.glob(os.path.join(data_dir2, "*.npz")))
    
    print(f"Found {len(p1_files)} P1_WIN files and {len(p2_files)} P2_WIN files")
    
    # Shuffle each group separately
    random.shuffle(p1_files)
    random.shuffle(p2_files)
    
    # Split each group proportionally
    num_p1_train = int(len(p1_files) * split)
    num_p2_train = int(len(p2_files) * split)
    
    train_files = p1_files[:num_p1_train] + p2_files[:num_p2_train]
    val_files = p1_files[num_p1_train:] + p2_files[num_p2_train:]
    
    # Shuffle the combined train and val sets
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    print(f"Train split: {num_p1_train} P1_WIN + {num_p2_train} P2_WIN = {len(train_files)} total")
    print(f"Val split: {len(p1_files) - num_p1_train} P1_WIN + {len(p2_files) - num_p2_train} P2_WIN = {len(val_files)} total")

    # Copy train files
    print(f"Copying {len(train_files)} training files to {train_dir}")
    for f in tqdm(train_files, desc="Copying train files"):
        shutil.copy(f, os.path.join(train_dir, os.path.basename(f)))

    # Copy val files
    print(f"Copying {len(val_files)} validation files to {val_dir}")
    for f in tqdm(val_files, desc="Copying val files"):
        shutil.copy(f, os.path.join(val_dir, os.path.basename(f)))

    print("\n--- Data Splitting Complete ---")
    print(f"Train data saved in: {train_dir}")
    print(f"Validation data saved in: {val_dir}")


def get_frames_from_npz(file_path):
    """
    Extracts frames from a .npz file and returns them as a tensor.
    """
    data = np.load(file_path)
    mask = data['valid_frames']
    idx = int(np.where(mask == 1)[0][-1])
    images = data['images'][:idx]
    video_tensor = torch.from_numpy(images).float()
    data.close()
    return video_tensor  # [t, c, h, w]


def detect_cuda_devices():
    """
    Detect the number of available CUDA devices on the system.
    
    Returns:
        int: Number of available CUDA devices. Returns 0 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return 0
    
    num_devices = torch.cuda.device_count()
    print(f"Detected {num_devices} CUDA device(s):")
    
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {device_name} ({memory_gb:.1f} GB)")
    
    return num_devices


if __name__ == "__main__":
    # Set multiprocessing start method at the very beginning
    mp.set_start_method('spawn', force=True)
    
    # Detect available CUDA devices
    available_gpus = detect_cuda_devices()
    
    # Configuration
    num_gpus = min(available_gpus, 8)  # Use detected GPUs, max 8
    if num_gpus == 0:
        print("No CUDA devices available. Exiting.")
        exit(1)
    
    batch_size = 2  # Conservative batch size to avoid OOM (was 0)
    
    print(f"Using {num_gpus} out of {available_gpus} available GPU(s)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wan_path = "preproccessing/checkpoints/Wan2.1/vae"
    ltx_path = "preproccessing/checkpoints/LTXV/vae"

    data1 = 'preproccessing/tekken_dataset_npz/P1_WIN'
    data2 = 'preproccessing/tekken_dataset_npz/P2_WIN'
    split_output_dir = "preproccessing/data_v3"
    
    # Create train/val split
    create_split(data1, data2, split_output_dir, split=0.9)
    
    # Process data with multi-GPU support
    # cache_dir = "preproccessing/cached_data_v3_wan"
    cache_dir_ltx = "preproccessing/cached_data_v3_ltx"
    '''process_data(
        data_dir=split_output_dir,
        output_dir=cache_dir,
        vae_ckpt_dir=wan_path,
        enable_tiling=False,
        num_gpus=num_gpus,
        batch_size=batch_size
    )'''
    process_data(
        data_dir=split_output_dir,
        output_dir=cache_dir_ltx,
        vae_ckpt_dir=ltx_path,
        enable_tiling=False,
        num_gpus=num_gpus,
        batch_size=batch_size
    )
