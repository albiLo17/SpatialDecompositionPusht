#!/usr/bin/env python3
"""Script to visualize dataset sequences and reference frames for debugging.

This script loads the segmented dataset and visualizes:
- Initial state of the environment
- Ground truth reference position (from dataset)
- Ground truth action sequence
- Observation sequence

This helps debug what reference frames are being used for training.

Usage:
    python SD_pusht/scripts/visualize_dataset_reference_frames.py \
        --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
        --output-dir debug_visualizations \
        --num-samples 80 \
        --pred-horizon 16 \
        --obs-horizon 2 \
        --action-horizon 8
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import imageio.v3 as iio

from SD_pusht.datasets import PushTSegmentedDataset, PushTSegmentedDatasetSimple
from SD_pusht.utils.environment import make_env, apply_legacy_state
from SD_pusht.utils.normalization import unnormalize_data


def visualize_dataset_sample(
    dataset,
    stats,
    sample_idx,
    out_path,
    num_obs_to_show=5,
):
    """
    Visualize a single dataset sample with reference frame and actions.
    
    Args:
        dataset: PushTSegmentedDataset instance
        stats: Dataset statistics for unnormalization
        sample_idx: Index of the sample to visualize
        out_path: Path to save the visualization
        num_obs_to_show: Number of observations to show in the sequence
    """
    # Get sample from dataset
    sample = dataset[sample_idx]
    obs_seq = sample['obs']  # (obs_horizon, obs_dim)
    actions = sample['action']  # (pred_horizon, act_dim)
    ref_pos = sample['reference_pos']  # (2,)
    
    # Get the indices to access the raw dataset
    indices_row = dataset.indices[sample_idx]
    
    # Handle different index structures
    if len(indices_row) == 6:
        # Original PushTSegmentedDataset: [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, seg_idx, ref_pos_idx]
        (buffer_start_idx, buffer_end_idx,
         sample_start_idx, sample_end_idx,
         seg_idx, ref_pos_idx) = indices_row
    else:
        # PushTSegmentedDatasetSimple: [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
        (buffer_start_idx, buffer_end_idx,
         sample_start_idx, sample_end_idx) = indices_row
        # Get reference frame index from the map
        if hasattr(dataset, 'reference_frame_map'):
            ref_pos_idx = dataset.reference_frame_map[sample_idx]
        else:
            # Fallback: use buffer_end_idx - 1
            ref_pos_idx = buffer_end_idx - 1
    
    # Get initial state (first observation)
    initial_state = obs_seq[0]  # (obs_dim,)
    
    # Get the actual state at the reference position index for comparison
    # This is the state at the END of the segment where ref_pos comes from
    # Clamp ref_pos_idx to valid range
    max_idx = len(dataset.normalized_train_data['obs']) - 1
    ref_pos_idx = max(0, min(ref_pos_idx, max_idx))
    ref_state_normalized = dataset.normalized_train_data['obs'][ref_pos_idx]
    ref_state_unnorm = unnormalize_data(ref_state_normalized[None, :], stats["obs"])[0]
    ref_state_agent_pos = ref_state_unnorm[:2]  # Agent position at ref_pos_idx
    
    # Unnormalize initial state for rendering
    initial_state_unnorm = unnormalize_data(initial_state[None, :], stats["obs"])[0]
    # State: [agent_x, agent_y, block_x, block_y, block_angle]
    
    # Unnormalize reference position
    agent_stats = {
        'min': stats["obs"]['min'][0:2],
        'max': stats["obs"]['max'][0:2],
    }
    ref_pos_unnorm = unnormalize_data(ref_pos[None, :], agent_stats)[0]
    
    # Debug: Print positions to check
    print(f"Sample {sample_idx}:")
    print(f"  Initial agent pos (from obs_seq[0]): {initial_state_unnorm[:2]}")
    print(f"  Reference pos (from dataset): {ref_pos_unnorm}")
    print(f"  Agent pos at ref_pos_idx (actual state): {ref_state_agent_pos}")
    print(f"  ref_pos_idx: {ref_pos_idx}, buffer_start_idx: {buffer_start_idx}")
    
    # Unnormalize actions
    action_stats = stats.get("action", None)
    if action_stats is not None:
        actions_unnorm = unnormalize_data(actions, action_stats)
    else:
        actions_unnorm = actions
    
    # Create environment and render initial state
    env_fn = make_env(render_mode="rgb_array", legacy=True)
    env = env_fn()
    env.reset(options={"reset_to_state": initial_state_unnorm})
    frame = env.render()  # (H, W, 3) uint8
    
    # Get the actual agent position from the environment after rendering
    # This will help us verify the coordinate mapping
    actual_agent_pos_from_env = np.array(env.agent.position)
    print(f"  Actual agent pos from env.agent.position: {actual_agent_pos_from_env}")
    print(f"  Initial state agent pos (from obs): {initial_state_unnorm[:2]}")
    print(f"  Frame shape: {frame.shape}")
    
    # Resize frame if needed
    if frame.shape[0] != 512 or frame.shape[1] != 512:
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Helper function to convert world coordinates to pixel coordinates
    # PushT uses [0, 512] for both x and y in world coordinates
    # The rendered image is 512x512
    # Try different mappings - the yellow circle will help verify which is correct
    def world_to_pixel(world_pos, swap_xy=False, flip_y=False):
        """Convert world coordinates [0, 512] to pixel coordinates [0, 512]
        
        Args:
            world_pos: [x, y] in world coordinates
            swap_xy: If True, swap x and y
            flip_y: If True, flip y-axis (for image coordinates where y=0 is at top)
        """
        if swap_xy:
            x_world, y_world = world_pos[1], world_pos[0]
        else:
            x_world, y_world = world_pos[0], world_pos[1]
        
        x_pixel = int((x_world / 512.0) * 512)
        if flip_y:
            y_pixel = int((1.0 - y_world / 512.0) * 512)  # Flip y-axis
        else:
            y_pixel = int((y_world / 512.0) * 512)
        return (x_pixel, y_pixel)
    
    # Try different coordinate mappings - check which one makes yellow circle match rendered agent
    # Default: no swap, flip y
    world_to_pixel_fn = lambda pos: world_to_pixel(pos, swap_xy=False, flip_y=True)
    
    # Draw actual agent position from environment (yellow) - to verify coordinate system
    actual_agent_pixel = world_to_pixel(actual_agent_pos_from_env)
    cv2.circle(frame_bgr, actual_agent_pixel, 8, (0, 255, 255), -1)  # Yellow filled circle
    cv2.putText(frame_bgr, "Env Agent", 
                (actual_agent_pixel[0] + 15, actual_agent_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw ground truth reference position (green, large) - from dataset
    ref_pos_pixel = world_to_pixel(ref_pos_unnorm)
    cv2.circle(frame_bgr, ref_pos_pixel, 15, (0, 255, 0), -1)  # Green filled circle
    cv2.circle(frame_bgr, ref_pos_pixel, 15, (0, 200, 0), 3)  # Green outline
    cv2.putText(frame_bgr, "GT Reference (from dataset)", 
                (ref_pos_pixel[0] + 20, ref_pos_pixel[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw actual agent position at ref_pos_idx (magenta) - for comparison
    ref_state_agent_pixel = world_to_pixel(ref_state_agent_pos)
    cv2.circle(frame_bgr, ref_state_agent_pixel, 12, (255, 0, 255), -1)  # Magenta filled circle
    cv2.circle(frame_bgr, ref_state_agent_pixel, 12, (200, 0, 200), 2)  # Magenta outline
    cv2.putText(frame_bgr, "Agent at ref_idx", 
                (ref_state_agent_pixel[0] + 20, ref_state_agent_pixel[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw line between them if they don't match
    if np.linalg.norm(ref_pos_unnorm - ref_state_agent_pos) > 1.0:
        cv2.line(frame_bgr, ref_pos_pixel, ref_state_agent_pixel, (0, 0, 255), 2)  # Red line if mismatch
        cv2.putText(frame_bgr, "MISMATCH!", 
                    ((ref_pos_pixel[0] + ref_state_agent_pixel[0]) // 2,
                     (ref_pos_pixel[1] + ref_state_agent_pixel[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw initial agent position (cyan)
    initial_agent_pos = initial_state_unnorm[:2]
    initial_agent_pixel = world_to_pixel(initial_agent_pos)
    cv2.circle(frame_bgr, initial_agent_pixel, 10, (255, 255, 0), -1)  # Cyan filled circle
    cv2.putText(frame_bgr, "Initial Agent", 
                (initial_agent_pixel[0] + 15, initial_agent_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw action sequence with colors (blue to yellow gradient)
    action_horizon = len(actions_unnorm)
    try:
        cmap = plt.colormaps['viridis']  # Color map from blue to yellow
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('viridis')  # Fallback for older matplotlib
    
    for t in range(action_horizon):
        # Get color based on timestep
        color_norm = t / max(action_horizon - 1, 1)
        color_rgb = cmap(color_norm)[:3]
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])
        
        # Action is target agent position
        action_pos = actions_unnorm[t]
        action_pixel = world_to_pixel(action_pos)
        
        # Draw circle for action
        radius = 6
        cv2.circle(frame_bgr, action_pixel, radius, color_bgr, -1)
        
        # Draw line from previous action (if not first)
        if t > 0:
            prev_action_pos = actions_unnorm[t-1]
            prev_action_pixel = world_to_pixel(prev_action_pos)
            cv2.line(frame_bgr, prev_action_pixel, action_pixel, color_bgr, 2)
        
        # Draw timestep number
        cv2.putText(frame_bgr, str(t), 
                    (action_pixel[0] + 8, action_pixel[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw line from initial agent position to first action
    first_action_pixel = world_to_pixel(actions_unnorm[0])
    cv2.line(frame_bgr, initial_agent_pixel, first_action_pixel, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw line from initial agent to reference position (dashed effect)
    for i in range(0, 20):
        t = i / 20.0
        x = int(initial_agent_pixel[0] * (1 - t) + ref_pos_pixel[0] * t)
        y = int(initial_agent_pixel[1] * (1 - t) + ref_pos_pixel[1] * t)
        if i % 3 == 0:  # Create dashed effect
            cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)
    
    # No text overlay - keep image clean
    
    # Convert back to RGB for saving
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.imsave(out_path, frame_rgb)
    
    env.close()
    
    return {
        "image_path": out_path,
        "ref_pos": ref_pos_unnorm,
        "initial_agent_pos": initial_agent_pos,
        "actions": actions_unnorm,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset reference frames")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to zarr dataset file")
    parser.add_argument("--output-dir", type=str, default="debug_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of samples to visualize")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="Starting index for visualization")
    parser.add_argument("--pred-horizon", type=int, default=16,
                       help="Prediction horizon")
    parser.add_argument("--obs-horizon", type=int, default=2,
                       help="Observation horizon")
    parser.add_argument("--action-horizon", type=int, default=8,
                       help="Action horizon")
    parser.add_argument("--max-demos", type=int, default=None,
                       help="Maximum number of demos to use")
    parser.add_argument("--contact-threshold", type=float, default=0.1,
                       help="Contact threshold for segmentation")
    parser.add_argument("--min-segment-length", type=int, default=5,
                       help="Minimum segment length")
    parser.add_argument("--use-original-dataset", action="store_true",
                       help="Use PushTSegmentedDataset (original) instead of PushTSegmentedDatasetSimple")
    
    args = parser.parse_args()
    
    # Create dataset
    print("Loading dataset...")
    if args.use_original_dataset:
        print("Using PushTSegmentedDataset (original version)")
        dataset = PushTSegmentedDataset(
            dataset_path=args.dataset_path,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            max_demos=args.max_demos,
            use_contact_segmentation=True,
            contact_threshold=args.contact_threshold,
            min_segment_length=args.min_segment_length,
        )
    else:
        print("Using PushTSegmentedDatasetSimple (simplified version)")
        dataset = PushTSegmentedDatasetSimple(
            dataset_path=args.dataset_path,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            max_demos=args.max_demos,
            use_contact_segmentation=True,
            contact_threshold=args.contact_threshold,
            min_segment_length=args.min_segment_length,
        )
    
    # Get stats from dataset
    stats = dataset.stats
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Visualizing {args.num_samples} samples starting from index {args.start_idx}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize samples
    image_paths = []
    for i in tqdm(range(args.num_samples), desc="Creating visualizations"):
        sample_idx = args.start_idx + i
        if sample_idx >= len(dataset):
            print(f"Warning: Sample index {sample_idx} exceeds dataset size {len(dataset)}")
            break
        
        out_path = os.path.join(args.output_dir, f"sample_{sample_idx:05d}.png")
        
        try:
            vis_results = visualize_dataset_sample(
                dataset=dataset,
                stats=stats,
                sample_idx=sample_idx,
                out_path=out_path,
            )
            image_paths.append(vis_results['image_path'])
        except Exception as e:
            print(f"Error visualizing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create GIF/video from saved images
    if image_paths:
        print(f"\nCreating animation from {len(image_paths)} saved images...")
        gif_path = os.path.join(args.output_dir, "samples_animation.gif")
        video_path = os.path.join(args.output_dir, "samples_animation.mp4")
        
        # Read all saved images
        frames = []
        for img_path in sorted(image_paths):  # Sort to ensure correct order
            if os.path.exists(img_path):
                img = iio.imread(img_path)
                frames.append(img)
        
        if frames:
            # Save as GIF
            iio.imwrite(gif_path, frames, duration=0.5, loop=0)  # 0.5 seconds per frame
            print(f"GIF saved to: {gif_path}")
            
            # Also save as MP4 video
            iio.imwrite(video_path, frames, fps=2)  # 2 fps = 0.5 seconds per frame
            print(f"Video saved to: {video_path}")
        else:
            print("Warning: No valid images found for animation creation")
    
    print(f"\nVisualizations saved to: {args.output_dir}")
    print(f"Total samples visualized: {len(image_paths)}")


if __name__ == "__main__":
    main()

