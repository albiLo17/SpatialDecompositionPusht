#!/usr/bin/env python3
"""
Unified script for rendering segmented PushT trajectories.

This script provides multiple rendering modes:
  - single: Render a single segment with environment visualization
  - gif: Create animated GIF with trajectory overlay
  - comparison: Create side-by-side comparison of multiple segments
  - batch-gif: Create GIFs for multiple segments

Features:
  - Loads full states from original zarr dataset
  - Trajectory visualization with gradient coloring
  - Contact/no-contact segment differentiation
  - Adjustable playback speed and frame sampling
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add paths for imports
REPO_ROOT = "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht"
SPATIALDECOMP_PATH = os.path.join(REPO_ROOT, "spatialdecomposition")
if SPATIALDECOMP_PATH not in sys.path:
    sys.path.append(SPATIALDECOMP_PATH)

sys.path.extend([
    "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht/libero",
    "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht/robomimic",
    "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht/robosuite",
    "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht/diffusion_policy",
])

from SpatialDecomposition.TwoD_table_play.data import ToyDataset
from gym_pusht.envs.pusht import PushTEnv
from PIL import Image, ImageDraw
import zarr


# ============================================================================
# Utility Functions
# ============================================================================

def apply_legacy_state(env, state):
    """Apply a state to the environment."""
    agent_pos = list(state[:2])
    block_pos = list(state[2:4])
    block_angle = float(state[4])

    env.agent.position = agent_pos
    env.block.position = block_pos
    env.block.angle = block_angle

    dt = getattr(env, "dt", 1.0 / getattr(env, "sim_hz", 100))
    env.space.step(dt)


def load_segment_states(original_dataset_path, segment_metadata):
    """
    Load the full states for a segment from the original zarr dataset.
    
    Args:
        original_dataset_path: Path to the original zarr dataset
        segment_metadata: Metadata dict for the segment
        
    Returns:
        states: Array of full states [T, 5] (agent_x, agent_y, block_x, block_y, block_angle)
    """
    ds = zarr.open(original_dataset_path, 'r')
    episode_ends = ds['meta']['episode_ends'][:]
    
    episode_idx = segment_metadata['original_episode']
    seg_start = segment_metadata['original_start']
    seg_end = segment_metadata['original_end']
    
    # Calculate global indices
    episode_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    global_start = episode_start + seg_start
    global_end = episode_start + seg_end
    
    # Load states
    states = ds['data']['state'][global_start:global_end]
    return states


def world_to_pixel(pos, env_size=512, world_min=0, world_max=1):
    """Convert world coordinates to pixel coordinates."""
    x = int((pos[0] - world_min) / (world_max - world_min) * env_size)
    y = int((1 - (pos[1] - world_min) / (world_max - world_min)) * env_size)  # Flip Y
    return (x, y)


# ============================================================================
# Rendering Mode: Single Segment
# ============================================================================

def render_single_segment(segment_states, segment_metadata, segment_idx,
                         render_mode="human", speed=1.0, save_images=False, 
                         output_dir=None):
    """
    Render a single segment with environment visualization.
    
    Args:
        segment_states: Array of full states [T, 5]
        segment_metadata: Metadata dict for this segment
        segment_idx: Index of the segment
        render_mode: "rgb_array" or "human"
        speed: Playback speed multiplier (only for human mode)
        save_images: Whether to save rendered images
        output_dir: Directory to save images (if save_images=True)
    
    Returns:
        images: List of rendered frames (if render_mode="rgb_array")
    """
    env = PushTEnv(render_mode=render_mode, legacy=True)
    
    # Reset to first state of the segment
    options = {"reset_to_state": segment_states[0]}
    obs, info = env.reset(options=options)
    
    print(f"Rendering segment {segment_idx} ({'Contact' if segment_metadata['contact_flag'] else 'No-Contact'})")
    print(f"  Episode: {segment_metadata['original_episode']}, Length: {len(segment_states)}")
    
    # Calculate timing for human mode
    dt = 1.0 / env.metadata.get("render_fps", 10)
    delay = dt / max(1.0, speed)
    
    images = []
    
    for t, state in enumerate(segment_states):
        # Apply the state to the environment
        apply_legacy_state(env, state)
        
        # Render the current state
        if render_mode == "rgb_array":
            img = env.render()
            images.append(img)
        else:
            env.render()
            time.sleep(delay)
        
        # Print progress for long segments
        if len(segment_states) > 50 and t % 10 == 0:
            print(f"  Progress: {t}/{len(segment_states)}")
    
    env.close()
    
    # Save images if requested
    if save_images and output_dir and render_mode == "rgb_array":
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for t, img in enumerate(images):
            img_path = output_dir / f"segment_{segment_idx:03d}_frame_{t:03d}.png"
            plt.imsave(img_path, img)
        
        print(f"  Saved {len(images)} images to {output_dir}")
    
    return images


# ============================================================================
# Rendering Mode: Animated GIF
# ============================================================================

def create_segment_gif(segment_idx, segment_metadata, original_dataset_path,
                      output_path, fps=10, trajectory_length=30, downsample=1,
                      show_block_trajectory=False):
    """
    Create an animated GIF for a segment with trajectory overlay.
    
    Args:
        segment_idx: Index of the segment
        segment_metadata: Metadata for all segments
        original_dataset_path: Path to original zarr dataset
        output_path: Path to save the GIF
        fps: Frames per second
        trajectory_length: Number of past positions to show
        downsample: Use every Nth frame (1 = all frames)
        show_block_trajectory: Also show block trajectory
    
    Returns:
        output_path: Path where GIF was saved
    """
    meta = segment_metadata[segment_idx]
    
    # Load full states
    states = load_segment_states(original_dataset_path, meta)
    
    print(f"Creating GIF for segment {segment_idx}")
    print(f"  Type: {'Contact' if meta['contact_flag'] else 'No-Contact'}")
    print(f"  Episode: {meta['original_episode']}")
    print(f"  Full Length: {len(states)} frames")
    print(f"  Downsample: {downsample}")
    
    # Create environment
    env = PushTEnv(render_mode="rgb_array", legacy=True)
    
    # Reset to first state
    options = {"reset_to_state": states[0]}
    obs, info = env.reset(options=options)
    
    frames = []
    agent_trajectory = []
    block_trajectory = []
    
    for t in range(0, len(states), downsample):
        state = states[t]
        
        # Apply the state
        apply_legacy_state(env, state)
        
        # Render
        img = env.render()
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Track positions
        agent_pos = state[:2]
        block_pos = state[2:4]
        agent_trajectory.append(agent_pos)
        block_trajectory.append(block_pos)
        
        # Keep only recent trajectory
        if len(agent_trajectory) > trajectory_length:
            agent_trajectory = agent_trajectory[-trajectory_length:]
            block_trajectory = block_trajectory[-trajectory_length:]
        
        # Draw agent trajectory
        if len(agent_trajectory) > 1:
            for i in range(len(agent_trajectory) - 1):
                p1 = world_to_pixel(agent_trajectory[i])
                p2 = world_to_pixel(agent_trajectory[i + 1])
                
                # Color gradient from old (faint) to new (bright)
                alpha = int(255 * (i + 1) / len(agent_trajectory))
                
                # Different colors for contact vs no-contact
                if meta['contact_flag']:
                    color = (255, alpha, alpha)  # Red gradient for contact
                else:
                    color = (alpha, alpha, 255)  # Blue gradient for no-contact
                
                draw.line([p1, p2], fill=color, width=3)
            
            # Draw current agent position
            current_pixel = world_to_pixel(agent_trajectory[-1])
            draw.ellipse([current_pixel[0]-5, current_pixel[1]-5, 
                         current_pixel[0]+5, current_pixel[1]+5],
                        fill=(255, 255, 0), outline=(0, 0, 0), width=2)
        
        # Draw block trajectory if requested
        if show_block_trajectory and len(block_trajectory) > 1:
            for i in range(len(block_trajectory) - 1):
                p1 = world_to_pixel(block_trajectory[i])
                p2 = world_to_pixel(block_trajectory[i + 1])
                
                # Green gradient for block
                alpha = int(255 * (i + 1) / len(block_trajectory))
                color = (alpha, 255, alpha)
                
                draw.line([p1, p2], fill=color, width=2)
            
            # Draw current block position
            current_pixel = world_to_pixel(block_trajectory[-1])
            draw.ellipse([current_pixel[0]-4, current_pixel[1]-4, 
                         current_pixel[0]+4, current_pixel[1]+4],
                        fill=(0, 255, 0), outline=(0, 0, 0), width=2)
        
        # Add text overlay
        contact_type = "CONTACT" if meta['contact_flag'] else "NO CONTACT"
        draw.text((10, 10), f"Segment {segment_idx} ({contact_type})", 
                 fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((10, 30), f"Frame {t}/{len(states)}", 
                 fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        draw.text((10, 50), f"Episode {meta['original_episode']}", 
                 fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        
        frames.append(pil_img)
        
        if t % 10 == 0:
            print(f"  Progress: {t}/{len(states)}")
    
    env.close()
    
    # Save as GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )
    
    print(f"✓ GIF saved to: {output_path}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Duration: {len(frames)/fps:.1f}s @ {fps} fps")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return str(output_path)


# ============================================================================
# Rendering Mode: Comparison Grid
# ============================================================================

def create_comparison_grid(segment_metadata, original_dataset_path, segment_indices,
                          save_path=None, frame_interval=1):
    """
    Create a comparison visualization showing multiple segments side-by-side.
    
    Args:
        segment_metadata: List of metadata for each segment
        original_dataset_path: Path to original zarr dataset
        segment_indices: List of segment indices to render
        save_path: Path to save the comparison image
        frame_interval: Render every Nth frame for sampling
    """
    n_segments = len(segment_indices)
    n_cols = min(3, n_segments)
    n_rows = (n_segments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_segments == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    print(f"Creating comparison grid for {n_segments} segments...")
    
    for i, seg_idx in enumerate(segment_indices):
        if i >= len(axes):
            break
            
        meta = segment_metadata[seg_idx]
        
        try:
            # Load full states for this segment
            states = load_segment_states(original_dataset_path, meta)
            
            # Render just the first frame
            env = PushTEnv(render_mode="rgb_array", legacy=True)
            options = {"reset_to_state": states[0]}
            obs, info = env.reset(options=options)
            apply_legacy_state(env, states[0])
            img = env.render()
            env.close()
            
            # Show the first frame
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            ax.imshow(img)
            
            # Add title with segment info
            contact_type = "Contact" if meta['contact_flag'] else "No-Contact"
            ax.set_title(f"Segment {seg_idx} ({contact_type})\n"
                       f"Episode {meta['original_episode']}, "
                       f"Length: {meta['segment_length']}")
            ax.axis('off')
            
            print(f"  ✓ Rendered segment {seg_idx}")
            
        except Exception as e:
            print(f"  ✗ Error rendering segment {seg_idx}: {e}")
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            ax.text(0.5, 0.5, f"Error loading\nsegment {seg_idx}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Segment {seg_idx} (Error)")
            ax.axis('off')
    
    # Hide unused subplots
    if isinstance(axes, (list, np.ndarray)):
        for i in range(n_segments, len(axes)):
            axes[i].set_visible(False)
    
    plt.suptitle(f"Segmented Trajectories Comparison ({n_segments} segments)", 
                fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison grid saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified script for rendering segmented PushT trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render a single segment interactively
  python render_segments.py single --segment-index 0
  
  # Create an animated GIF with trajectory overlay
  python render_segments.py gif --segment-index 0 --output segment_0.gif
  
  # Create comparison grid of multiple segments
  python render_segments.py comparison --segment-indices 0 1 2 3 --save-path comparison.png
  
  # Batch create GIFs for contact segments only
  python render_segments.py batch-gif --contact-only --max-segments 5
        """
    )
    
    # Subcommands for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Rendering mode')
    
    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--segmented-dataset", 
                           default="datasets/pusht_segmented_clean_fixed.npz",
                           help="Path to segmented dataset")
    common_args.add_argument("--metadata", 
                           default="datasets/pusht_segmented_clean_metadata.npz",
                           help="Path to segmentation metadata")
    common_args.add_argument("--original-dataset",
                           default="datasets/pusht_cchi_v7_replay.zarr.zip",
                           help="Path to original zarr dataset")
    
    # Single segment rendering
    single_parser = subparsers.add_parser('single', parents=[common_args],
                                         help='Render a single segment')
    single_parser.add_argument("--segment-index", type=int, required=True,
                              help="Segment index to render")
    single_parser.add_argument("--render-mode", choices=["rgb_array", "human"], 
                              default="human", help="Rendering mode")
    single_parser.add_argument("--speed", type=float, default=1.0,
                              help="Playback speed multiplier (human mode only)")
    single_parser.add_argument("--save-images", action="store_true",
                              help="Save rendered images to disk")
    single_parser.add_argument("--output-dir", default="rendered_segments",
                              help="Output directory for saved images")
    
    # GIF creation
    gif_parser = subparsers.add_parser('gif', parents=[common_args],
                                      help='Create animated GIF for a segment')
    gif_parser.add_argument("--segment-index", type=int, required=True,
                           help="Segment index to render")
    gif_parser.add_argument("--output", required=True,
                           help="Output path for GIF file")
    gif_parser.add_argument("--fps", type=int, default=10,
                           help="Frames per second")
    gif_parser.add_argument("--trajectory-length", type=int, default=30,
                           help="Number of past positions to show in trajectory")
    gif_parser.add_argument("--downsample", type=int, default=1,
                           help="Use every Nth frame (1 = all frames)")
    gif_parser.add_argument("--show-block-trajectory", action="store_true",
                           help="Also show block trajectory")
    
    # Comparison grid
    comp_parser = subparsers.add_parser('comparison', parents=[common_args],
                                       help='Create comparison grid of segments')
    comp_parser.add_argument("--segment-indices", nargs="+", type=int,
                            help="Specific segment indices to render")
    comp_parser.add_argument("--max-segments", type=int, default=6,
                            help="Maximum number of segments (if indices not specified)")
    comp_parser.add_argument("--save-path",
                            help="Path to save comparison image (if not provided, shows interactively)")
    comp_parser.add_argument("--contact-only", action="store_true",
                            help="Only show contact segments")
    comp_parser.add_argument("--no-contact-only", action="store_true",
                            help="Only show no-contact segments")
    
    # Batch GIF creation
    batch_parser = subparsers.add_parser('batch-gif', parents=[common_args],
                                        help='Create GIFs for multiple segments')
    batch_parser.add_argument("--segment-indices", nargs="+", type=int,
                             help="Specific segment indices to render")
    batch_parser.add_argument("--max-segments", type=int, default=4,
                             help="Maximum number of segments")
    batch_parser.add_argument("--output-dir", default="segment_gifs",
                             help="Output directory for GIFs")
    batch_parser.add_argument("--fps", type=int, default=10,
                             help="Frames per second")
    batch_parser.add_argument("--trajectory-length", type=int, default=30,
                             help="Number of past positions to show in trajectory")
    batch_parser.add_argument("--downsample", type=int, default=1,
                             help="Use every Nth frame (1 = all frames)")
    batch_parser.add_argument("--show-block-trajectory", action="store_true",
                             help="Also show block trajectory")
    batch_parser.add_argument("--contact-only", action="store_true",
                             help="Only create GIFs for contact segments")
    batch_parser.add_argument("--no-contact-only", action="store_true",
                             help="Only create GIFs for no-contact segments")
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    # Load segmented dataset
    print("Loading segmented dataset...")
    segmented_dataset = ToyDataset.from_file(args.segmented_dataset)
    metadata_data = np.load(args.metadata, allow_pickle=True)
    segment_metadata = metadata_data['segment_metadata']
    print(f"Loaded {len(segmented_dataset)} segments\n")
    
    # Execute based on mode
    if args.mode == 'single':
        # Validate segment index
        if args.segment_index >= len(segment_metadata):
            print(f"Error: segment_index {args.segment_index} is out of range (max: {len(segment_metadata)-1})")
            return
        
        meta = segment_metadata[args.segment_index]
        states = load_segment_states(args.original_dataset, meta)
        
        render_single_segment(
            states, meta, args.segment_index,
            render_mode=args.render_mode,
            speed=args.speed,
            save_images=args.save_images,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'gif':
        # Validate segment index
        if args.segment_index >= len(segment_metadata):
            print(f"Error: segment_index {args.segment_index} is out of range (max: {len(segment_metadata)-1})")
            return
        
        create_segment_gif(
            segment_idx=args.segment_index,
            segment_metadata=segment_metadata,
            original_dataset_path=args.original_dataset,
            output_path=args.output,
            fps=args.fps,
            trajectory_length=args.trajectory_length,
            downsample=args.downsample,
            show_block_trajectory=args.show_block_trajectory
        )
    
    elif args.mode == 'comparison':
        # Determine which segments to show
        if args.segment_indices:
            segment_indices = args.segment_indices
        else:
            # Filter by contact type if requested
            if args.contact_only:
                segment_indices = [i for i, m in enumerate(segment_metadata) if m['contact_flag']]
            elif args.no_contact_only:
                segment_indices = [i for i, m in enumerate(segment_metadata) if not m['contact_flag']]
            else:
                segment_indices = list(range(len(segment_metadata)))
            
            # Limit to max_segments
            segment_indices = segment_indices[:args.max_segments]
        
        create_comparison_grid(
            segment_metadata=segment_metadata,
            original_dataset_path=args.original_dataset,
            segment_indices=segment_indices,
            save_path=args.save_path
        )
    
    elif args.mode == 'batch-gif':
        # Determine which segments to render
        if args.segment_indices:
            segment_indices = args.segment_indices
        else:
            # Filter by contact type if requested
            if args.contact_only:
                segment_indices = [i for i, m in enumerate(segment_metadata) if m['contact_flag']]
            elif args.no_contact_only:
                segment_indices = [i for i, m in enumerate(segment_metadata) if not m['contact_flag']]
            else:
                segment_indices = list(range(len(segment_metadata)))
            
            # Limit to max_segments
            segment_indices = segment_indices[:args.max_segments]
        
        print(f"Creating GIFs for {len(segment_indices)} segments...\n")
        
        # Create GIFs
        output_paths = []
        for seg_idx in segment_indices:
            meta = segment_metadata[seg_idx]
            contact_type = "contact" if meta['contact_flag'] else "no_contact"
            output_path = Path(args.output_dir) / f"segment_{seg_idx:03d}_{contact_type}.gif"
            
            try:
                gif_path = create_segment_gif(
                    segment_idx=seg_idx,
                    segment_metadata=segment_metadata,
                    original_dataset_path=args.original_dataset,
                    output_path=output_path,
                    fps=args.fps,
                    trajectory_length=args.trajectory_length,
                    downsample=args.downsample,
                    show_block_trajectory=args.show_block_trajectory
                )
                output_paths.append(gif_path)
                print()
            except Exception as e:
                print(f"✗ Error creating GIF for segment {seg_idx}: {e}\n")
        
        print(f"{'='*60}")
        print(f"Created {len(output_paths)}/{len(segment_indices)} GIFs in {args.output_dir}/")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

