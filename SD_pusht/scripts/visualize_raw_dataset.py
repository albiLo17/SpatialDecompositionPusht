#!/usr/bin/env python3
"""Simple script to visualize the raw collected dataset.

This script loads a zarr dataset and creates simple visualizations showing:
- Agent trajectories for each episode
- State values over time
- Images if available

Example:
    python SD_pusht/scripts/visualize_raw_dataset.py \
        --dataset-path datasets/ropeflatten_expert.zarr \
        --output-dir visualizations/raw_dataset
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zarr
import imageio.v3 as iio


def visualize_episode_trajectory(
    root,
    episode_idx: int,
    output_path: Path,
):
    """Visualize a single episode's trajectory.
    
    Args:
        root: Zarr root group
        episode_idx: Index of episode to visualize
        output_path: Path to save visualization
    """
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # Get episode boundaries
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    # Extract episode data
    states = data['state'][start_idx:end_idx]
    actions = data['action'][start_idx:end_idx]
    
    # Get metadata
    agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
    env_state_dim = int(meta.attrs.get('env_state_dim', 0))
    
    # Extract agent positions (first 2 dims)
    agent_positions = states[:, :2]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Agent trajectory (top-left)
    ax1 = axes[0, 0]
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax1.scatter(agent_positions[0, 0], agent_positions[0, 1], 
               color='green', s=200, marker='o', label='Start', zorder=5)
    ax1.scatter(agent_positions[-1, 0], agent_positions[-1, 1], 
               color='red', s=200, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Episode {episode_idx} - Agent Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Agent X position over time (top-right)
    ax2 = axes[0, 1]
    time_steps = np.arange(len(agent_positions))
    ax2.plot(time_steps, agent_positions[:, 0], 'b-', linewidth=2, label='X Position')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('X Position')
    ax2.set_title('Agent X Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Agent Y position over time (bottom-left)
    ax3 = axes[1, 0]
    ax3.plot(time_steps, agent_positions[:, 1], 'r-', linewidth=2, label='Y Position')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Agent Y Position Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Action magnitude over time (bottom-right)
    ax4 = axes[1, 1]
    action_magnitude = np.linalg.norm(actions, axis=1)
    ax4.plot(time_steps, action_magnitude, 'g-', linewidth=2, label='Action Magnitude')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Action Magnitude')
    ax4.set_title('Action Magnitude Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved episode {episode_idx} visualization to: {output_path}")


def visualize_all_episodes(
    root,
    output_path: Path,
    max_episodes: int = 10,
):
    """Visualize trajectories for all episodes in one figure.
    
    Args:
        root: Zarr root group
        output_path: Path to save visualization
        max_episodes: Maximum number of episodes to show
    """
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    num_episodes = min(len(episode_ends), max_episodes)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
    
    for ep_idx in range(num_episodes):
        start_idx = episode_ends[ep_idx - 1] if ep_idx > 0 else 0
        end_idx = episode_ends[ep_idx]
        
        states = data['state'][start_idx:end_idx]
        agent_positions = states[:, :2]
        
        ax.plot(agent_positions[:, 0], agent_positions[:, 1], 
               color=colors[ep_idx], linewidth=2, alpha=0.7, 
               label=f'Episode {ep_idx}')
        ax.scatter(agent_positions[0, 0], agent_positions[0, 1], 
                  color=colors[ep_idx], s=100, marker='o', zorder=5)
        ax.scatter(agent_positions[-1, 0], agent_positions[-1, 1], 
                  color=colors[ep_idx], s=100, marker='s', zorder=5)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'All Episodes Trajectories (showing {num_episodes} episodes)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved all episodes visualization to: {output_path}")


def visualize_state_statistics(
    root,
    output_path: Path,
):
    """Visualize statistics of the state data.
    
    Args:
        root: Zarr root group
        output_path: Path to save visualization
    """
    data = root['data']
    meta = root['meta']
    
    states = data['state'][:]
    actions = data['action'][:]
    
    agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
    env_state_dim = int(meta.attrs.get('env_state_dim', 0))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: State value distributions (agent state)
    ax1 = axes[0, 0]
    if agent_state_dim > 0:
        agent_states = states[:, :agent_state_dim]
        for dim in range(min(agent_state_dim, 6)):  # Show first 6 dims
            ax1.hist(agent_states[:, dim], bins=50, alpha=0.5, label=f'Dim {dim}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Agent State Value Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: State value distributions (env state)
    ax2 = axes[0, 1]
    if env_state_dim > 0:
        env_states = states[:, agent_state_dim:agent_state_dim + env_state_dim]
        # Show first few dimensions
        for dim in range(min(env_state_dim, 6)):
            ax2.hist(env_states[:, dim], bins=50, alpha=0.5, label=f'Dim {dim}')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Environment State Value Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action value distributions
    ax3 = axes[1, 0]
    action_dim = actions.shape[1]
    for dim in range(min(action_dim, 6)):
        ax3.hist(actions[:, dim], bins=50, alpha=0.5, label=f'Dim {dim}')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Action Value Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Episode lengths
    ax4 = axes[1, 1]
    episode_ends = meta['episode_ends'][:]
    episode_lengths = []
    for i in range(len(episode_ends)):
        start_idx = episode_ends[i - 1] if i > 0 else 0
        end_idx = episode_ends[i]
        episode_lengths.append(end_idx - start_idx)
    
    ax4.hist(episode_lengths, bins=20, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Episode Length (frames)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Episode Length Distribution (mean: {np.mean(episode_lengths):.1f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved state statistics to: {output_path}")


def create_episode_gif(
    root,
    episode_idx: int,
    output_path: Path,
    fps: int = 5,
):
    """Create a GIF of all images from a single episode.
    
    Args:
        root: Zarr root group
        episode_idx: Index of episode to visualize
        output_path: Path to save GIF
        fps: Frames per second for the GIF
    """
    if 'image' not in root['data']:
        print("No images found in dataset, cannot create GIF")
        return
    
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # Get episode boundaries
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    # Extract episode images
    images = data['image'][start_idx:end_idx]
    states = data['state'][start_idx:end_idx]
    
    # Extract agent positions for overlay
    agent_positions = states[:, :2]
    
    print(f"  Creating GIF with {len(images)} frames...")
    
    frames = []
    for frame_idx in range(len(images)):
        img = images[frame_idx].copy()
        
        # Handle different image formats
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Add text overlay with frame info
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # Draw semi-transparent background for text
        text_bg = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_bg)
        text_draw.rectangle([5, 5, 300, 60], fill=(0, 0, 0, 180))
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), text_bg).convert('RGB')
        draw = ImageDraw.Draw(img_pil)
        
        # Add text
        draw.text((10, 10), f"Episode {episode_idx} - Frame {frame_idx}/{len(images)-1}", 
                 fill=(255, 255, 255), font=font)
        draw.text((10, 35), f"Pos: ({agent_positions[frame_idx, 0]:.2f}, {agent_positions[frame_idx, 1]:.2f})", 
                 fill=(255, 255, 255), font=font)
        
        frames.append(np.array(img_pil))
    
    # Save as GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, frames, duration=1.0/fps, loop=0)
    print(f"Saved episode {episode_idx} GIF to: {output_path}")


def visualize_sample_images(
    root,
    output_path: Path,
    num_samples: int = 16,
):
    """Visualize sample images from the dataset.
    
    Args:
        root: Zarr root group
        output_path: Path to save visualization
        num_samples: Number of sample images to show
    """
    if 'image' not in root['data']:
        print("No images found in dataset, skipping image visualization")
        return
    
    data = root['data']
    images = data['image']
    total_frames = images.shape[0]
    
    # Sample evenly across the dataset
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, sample_idx in enumerate(sample_indices):
        ax = axes[idx]
        img = images[sample_idx]
        
        # Handle different image formats
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        ax.imshow(img)
        ax.set_title(f'Frame {sample_idx}')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(sample_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample Images from Dataset (showing {num_samples} frames)', fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample images to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize raw collected dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to zarr dataset")
    parser.add_argument("--output-dir", type=str, default="visualizations/raw_dataset",
                       help="Output directory for visualizations")
    parser.add_argument("--num-episodes", type=int, default=5,
                       help="Number of individual episodes to visualize")
    parser.add_argument("--max-episodes-overview", type=int, default=10,
                       help="Maximum number of episodes in overview plot")
    parser.add_argument("--include-images", action="store_true",
                       help="Include image visualizations (if available)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    root = zarr.open(args.dataset_path, 'r')
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    num_episodes = len(episode_ends)
    
    print(f"Dataset loaded:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Total frames: {episode_ends[-1] if len(episode_ends) > 0 else 0}")
    print(f"  Agent state dim: {meta.attrs.get('agent_state_dim', 'N/A')}")
    print(f"  Env state dim: {meta.attrs.get('env_state_dim', 'N/A')}")
    print(f"  Action dim: {meta.attrs.get('action_dim', 'N/A')}")
    print(f"  Has images: {'image' in root['data']}")
    print()
    
    # Visualize all episodes overview
    print("Creating overview of all episodes...")
    visualize_all_episodes(
        root,
        output_dir / "all_episodes_overview.png",
        max_episodes=args.max_episodes_overview,
    )
    
    # Visualize individual episodes
    num_vis_episodes = min(args.num_episodes, num_episodes)
    print(f"\nVisualizing {num_vis_episodes} individual episodes...")
    for ep_idx in range(num_vis_episodes):
        visualize_episode_trajectory(
            root,
            ep_idx,
            output_dir / f"episode_{ep_idx}_trajectory.png",
        )
        # Create GIF of episode images
        if args.include_images:
            print(f"  Creating GIF for episode {ep_idx}...")
            create_episode_gif(
                root,
                ep_idx,
                output_dir / f"episode_{ep_idx}.gif",
                fps=5,
            )
    
    # Visualize statistics
    print("\nCreating state statistics visualization...")
    visualize_state_statistics(
        root,
        output_dir / "state_statistics.png",
    )
    
    # Visualize sample images
    if args.include_images:
        print("\nCreating sample images visualization...")
        visualize_sample_images(
            root,
            output_dir / "sample_images.png",
            num_samples=16,
        )
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
