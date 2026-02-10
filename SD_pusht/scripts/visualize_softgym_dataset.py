#!/usr/bin/env python3
"""Visualization script for SoftGym zarr dataset.

This script loads a SoftGym dataset and visualizes:
- Episode trajectories with images
- Segmentation boundaries
- Reference positions
- Sample batches

Example:
    python SD_pusht/scripts/visualize_softgym_dataset.py \
        --dataset-path datasets/ropeflatten_expert.zarr \
        --output-dir visualizations/ropeflatten \
        --num-episodes 5 \
        --num-samples 10
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import zarr
import torch
from torch.utils.data import DataLoader
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from SD_pusht.datasets.softgym_segmented_dataset import SoftGymSegmentedDatasetSimple


def visualize_episode(
    dataset_path: str,
    episode_idx: int,
    output_path: Path,
    include_images: bool = True,
):
    """Visualize a single episode with segmentation boundaries.
    
    Args:
        dataset_path: Path to zarr dataset
        episode_idx: Index of episode to visualize
        output_path: Path to save visualization
        include_images: Whether to include images in visualization
    """
    # Load dataset
    root = zarr.open(dataset_path, 'r')
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # Get episode boundaries
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    # Extract episode data
    states = data['state'][start_idx:end_idx]
    actions = data['action'][start_idx:end_idx]
    images = None
    if include_images and 'image' in data:
        images = data['image'][start_idx:end_idx]
    
    # Get metadata
    agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
    env_state_dim = int(meta.attrs.get('env_state_dim', 0))
    
    # Extract agent positions (first 2 dims of agent state)
    agent_positions = states[:, :2]
    
    # Create dataset to get segmentation
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        demo_indices=[episode_idx],
        use_contact_segmentation=True,
        contact_threshold=0.1,
        min_segment_length=5,
    )
    
    # Get segments for this episode
    demo_id = f"demo_{0}"  # Since we only loaded one episode, it's demo_0
    segments = dataset.segment_bounds.get(demo_id, [])
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    if include_images and images is not None:
        # Create grid of images with trajectory overlay
        num_frames = len(images)
        cols = min(8, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        for i in range(min(num_frames, rows * cols)):
            ax = plt.subplot(rows, cols, i + 1)
            
            # Display image
            img = images[i]
            if img.ndim == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            
            # Overlay agent position
            if i < len(agent_positions):
                pos = agent_positions[i]
                # Normalize position to image coordinates (assuming image is 128x128)
                # You may need to adjust this based on your environment
                img_h, img_w = img.shape[:2]
                x_norm = (pos[0] + 1) * img_w / 2  # Assuming normalized to [-1, 1]
                y_norm = (pos[1] + 1) * img_h / 2
                ax.plot(x_norm, y_norm, 'ro', markersize=8)
            
            # Mark segment boundaries
            for seg_start, seg_end in segments:
                if seg_start <= i < seg_end:
                    # Draw segment boundary
                    if i == seg_start:
                        ax.axvline(0, color='green', linewidth=3, alpha=0.7)
                    if i == seg_end - 1:
                        ax.axvline(img_w - 1, color='red', linewidth=3, alpha=0.7)
                    break
            
            ax.set_title(f'Frame {i}', fontsize=8)
            ax.axis('off')
        
        plt.suptitle(f'Episode {episode_idx} - Trajectory Visualization', fontsize=14, y=0.995)
        plt.tight_layout()
    else:
        # Plot trajectory without images
        ax = plt.subplot(1, 1, 1)
        
        # Plot agent trajectory
        ax.plot(agent_positions[:, 0], agent_positions[:, 1], 'b-', linewidth=2, label='Agent Trajectory')
        ax.scatter(agent_positions[:, 0], agent_positions[:, 1], c=range(len(agent_positions)), 
                  cmap='viridis', s=50, alpha=0.6)
        
        # Mark segment boundaries
        for seg_start, seg_end in segments:
            if seg_start < len(agent_positions) and seg_end <= len(agent_positions):
                start_pos = agent_positions[seg_start]
                end_pos = agent_positions[seg_end - 1]
                ax.scatter([start_pos[0]], [start_pos[1]], color='green', s=200, 
                          marker='s', label='Segment Start' if seg_start == segments[0][0] else '')
                ax.scatter([end_pos[0]], [end_pos[1]], color='red', s=200, 
                          marker='s', label='Segment End' if seg_end == segments[0][1] else '')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Episode {episode_idx} - Agent Trajectory with Segmentation')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved episode visualization to: {output_path}")


def visualize_segmentation(
    dataset_path: str,
    episode_idx: int,
    output_path: Path,
):
    """Visualize segmentation boundaries for an episode.
    
    Args:
        dataset_path: Path to zarr dataset
        episode_idx: Index of episode to visualize
        output_path: Path to save visualization
    """
    # Create dataset
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        demo_indices=[episode_idx],
        use_contact_segmentation=True,
        contact_threshold=0.1,
        min_segment_length=5,
    )
    
    # Load raw data
    root = zarr.open(dataset_path, 'r')
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    states = data['state'][start_idx:end_idx]
    agent_positions = states[:, :2]
    
    # Get segments
    demo_id = f"demo_{0}"
    segments = dataset.segment_bounds.get(demo_id, [])
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Trajectory with segment boundaries
    ax1 = axes[0]
    ax1.plot(agent_positions[:, 0], agent_positions[:, 1], 'b-', linewidth=2, alpha=0.5, label='Trajectory')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        seg_positions = agent_positions[seg_start:seg_end]
        ax1.plot(seg_positions[:, 0], seg_positions[:, 1], 'o-', 
                color=colors[seg_idx], linewidth=2, markersize=4, 
                label=f'Segment {seg_idx + 1}')
        # Mark boundaries
        ax1.scatter([seg_positions[0, 0]], [seg_positions[0, 1]], 
                   color=colors[seg_idx], s=200, marker='s', edgecolors='black', linewidths=2)
        if seg_end < len(agent_positions):
            ax1.scatter([agent_positions[seg_end - 1, 0]], [agent_positions[seg_end - 1, 1]], 
                       color=colors[seg_idx], s=200, marker='^', edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Episode {episode_idx} - Trajectory Segmentation')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Environment state movement over time
    ax2 = axes[1]
    env_state_start_idx = dataset.env_state_start_idx
    env_states = states[:, env_state_start_idx:]
    
    # Calculate movement
    if len(env_states) > 1:
        env_movement = np.linalg.norm(np.diff(env_states, axis=0), axis=1)
        env_movement = np.concatenate([[0], env_movement])
    else:
        env_movement = np.array([0])
    
    ax2.plot(env_movement, 'b-', linewidth=2, label='Environment State Movement')
    ax2.axhline(y=dataset.contact_threshold, color='r', linestyle='--', 
               linewidth=2, label=f'Threshold ({dataset.contact_threshold})')
    
    # Mark segment boundaries
    for seg_start, seg_end in segments:
        ax2.axvspan(seg_start, seg_end - 1, alpha=0.2, color='green')
        ax2.axvline(seg_start, color='green', linestyle='--', linewidth=1, alpha=0.7)
        if seg_end < len(env_movement):
            ax2.axvline(seg_end - 1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Movement Magnitude')
    ax2.set_title('Environment State Movement with Segmentation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved segmentation visualization to: {output_path}")


def visualize_gripper_refs_2d(
    dataset_path: str,
    episode_idx: int,
    output_path: Path,
):
    """Create a 2D plot showing gripper trajectories and reference frames.
    
    This is a simple 2D visualization to verify that reference frames are changing
    correctly, without projection issues.
    
    Args:
        dataset_path: Path to zarr dataset
        episode_idx: Index of episode to visualize
        output_path: Path to save plot
    """
    # Load dataset
    root = zarr.open(dataset_path, 'r')
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # Get episode boundaries
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    # Extract episode data
    states = data['state'][start_idx:end_idx]
    
    # Get metadata
    agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
    num_pickers = int(meta.attrs.get('num_picker', 2))
    
    # Extract picker positions (x, y, z for each picker)
    picker_positions = states[:, :agent_state_dim].reshape(len(states), num_pickers, 3)
    
    # Create dataset with gripper segmentation to get reference frames
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        demo_indices=[episode_idx],
        use_gripper_segmentation=True,
        use_contact_segmentation=False,
        min_segment_length=5,
    )
    
    # Build mapping from frame index to reference positions for each gripper
    # Get gripper segments for this episode
    demo_id = f"demo_{0}"  # Since we only loaded one episode, it's demo_0
    gripper_segment_bounds = dataset.gripper_segment_bounds.get(demo_id, {})
    
    # Debug: print segment information
    print(f"  Episode {episode_idx} - Gripper segments:")
    for gripper_id in range(num_pickers):
        segments = gripper_segment_bounds.get(gripper_id, [])
        print(f"    Gripper {gripper_id}: {len(segments)} segments")
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            print(f"      Segment {seg_idx}: frames {seg_start}-{seg_end}")
    
    # For each frame, find which segment it belongs to and get reference frame
    frame_to_ref_positions = {}
    for frame_idx in range(len(picker_positions)):
        ref_positions = []
        
        for gripper_id in range(num_pickers):
            gripper_segments = gripper_segment_bounds.get(gripper_id, [])
            
            if len(gripper_segments) == 0:
                # No segments, use end of episode
                ref_frame_idx = len(picker_positions) - 1
            else:
                # Find which segment this frame belongs to
                segment_idx = None
                for seg_idx, (seg_start, seg_end) in enumerate(gripper_segments):
                    if seg_start <= frame_idx < seg_end:
                        segment_idx = seg_idx
                        break
                
                if segment_idx is None:
                    # Frame doesn't fall in any segment, use end of episode
                    ref_frame_idx = len(picker_positions) - 1
                else:
                    # Check if there's a next segment for this gripper
                    if segment_idx + 1 < len(gripper_segments):
                        # Next segment exists, use its start as reference
                        next_seg_start, _ = gripper_segments[segment_idx + 1]
                        ref_frame_idx = next_seg_start
                    else:
                        # This is the last segment for this gripper
                        # Use end of episode as reference
                        ref_frame_idx = len(picker_positions) - 1
            
            # Get 3D position at reference frame
            ref_frame_idx = max(0, min(ref_frame_idx, len(picker_positions) - 1))
            ref_pos_3d = picker_positions[ref_frame_idx, gripper_id, :]
            ref_positions.append(ref_pos_3d)
        
        frame_to_ref_positions[frame_idx] = ref_positions
    
    # Create 2D plot (top-down view: x vs z)
    fig, axes = plt.subplots(1, num_pickers, figsize=(8 * num_pickers, 8))
    if num_pickers == 1:
        axes = [axes]
    
    colors = ['green', 'blue', 'magenta', 'yellow']
    
    for gripper_id in range(num_pickers):
        ax = axes[gripper_id]
        
        # Plot gripper trajectory (x, z coordinates for top-down view)
        gripper_trajectory = picker_positions[:, gripper_id, :]
        ax.plot(gripper_trajectory[:, 0], gripper_trajectory[:, 2], 
               'o-', color=colors[gripper_id], linewidth=2, markersize=4, 
               alpha=0.6, label='Gripper Trajectory')
        
        # Plot reference positions at each frame
        ref_x = []
        ref_z = []
        frame_indices = []
        for frame_idx in range(len(picker_positions)):
            ref_pos_3d = frame_to_ref_positions[frame_idx][gripper_id]
            ref_x.append(ref_pos_3d[0])
            ref_z.append(ref_pos_3d[2])
            frame_indices.append(frame_idx)
        
        # Plot reference positions
        ax.scatter(ref_x, ref_z, c=frame_indices, cmap='viridis', 
                  s=100, marker='*', edgecolors='black', linewidths=1,
                  label='Reference Frames', zorder=5)
        
        # Add colorbar to show frame progression
        scatter = ax.scatter(ref_x, ref_z, c=frame_indices, cmap='viridis', s=100)
        plt.colorbar(scatter, ax=ax, label='Frame Index')
        
        # Mark start and end
        ax.scatter([gripper_trajectory[0, 0]], [gripper_trajectory[0, 2]], 
                  color='green', s=200, marker='s', label='Start', zorder=6)
        ax.scatter([gripper_trajectory[-1, 0]], [gripper_trajectory[-1, 2]], 
                  color='red', s=200, marker='^', label='End', zorder=6)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title(f'Gripper {gripper_id} - Trajectory and Reference Frames\n(Episode {episode_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D reference frames plot to: {output_path}")


def visualize_trajectory_gif_with_gripper_refs(
    dataset_path: str,
    episode_idx: int,
    output_path: Path,
    fps: float = 0.5,
    include_images: bool = True,
):
    """Create a GIF showing the entire trajectory with gripper reference frames projected on images.
    
    Args:
        dataset_path: Path to zarr dataset
        episode_idx: Index of episode to visualize
        output_path: Path to save GIF
        fps: Frames per second for the GIF (default: 3 for slow viewing)
        include_images: Whether to use images from dataset (if available)
    """
    if not include_images or 'image' not in zarr.open(dataset_path, 'r')['data']:
        print("Images not available, cannot create visualization with projections")
        return
    
    # Load dataset
    root = zarr.open(dataset_path, 'r')
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # Get episode boundaries
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    # Extract episode data
    states = data['state'][start_idx:end_idx]
    actions = data['action'][start_idx:end_idx]
    images = data['image'][start_idx:end_idx]
    
    # Get metadata
    agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
    num_pickers = int(meta.attrs.get('num_picker', 2))
    
    # Extract picker positions (x, y, z for each picker)
    # Agent state is [x1, y1, z1, x2, y2, z2, ...]
    picker_positions = states[:, :agent_state_dim].reshape(len(states), num_pickers, 3)
    
    # Create dataset with gripper segmentation to get reference frames
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        demo_indices=[episode_idx],
        use_gripper_segmentation=True,
        use_contact_segmentation=False,
        min_segment_length=5,
    )
    
    # Build mapping from frame index to reference positions for each gripper
    # Get gripper segments for this episode
    demo_id = f"demo_{0}"  # Since we only loaded one episode, it's demo_0
    gripper_segment_bounds = dataset.gripper_segment_bounds.get(demo_id, {})
    
    # For each frame, find which segment it belongs to and get reference frame
    frame_to_ref_positions = {}
    for frame_idx in range(len(picker_positions)):
        ref_positions = []
        
        for gripper_id in range(num_pickers):
            gripper_segments = gripper_segment_bounds.get(gripper_id, [])
            
            if len(gripper_segments) == 0:
                # No segments, use end of episode
                ref_frame_idx = len(picker_positions) - 1
            else:
                # Find which segment this frame belongs to
                segment_idx = None
                for seg_idx, (seg_start, seg_end) in enumerate(gripper_segments):
                    if seg_start <= frame_idx < seg_end:
                        segment_idx = seg_idx
                        break
                
                if segment_idx is None:
                    # Frame doesn't fall in any segment, use end of episode
                    ref_frame_idx = len(picker_positions) - 1
                else:
                    # Check if there's a next segment for this gripper
                    if segment_idx + 1 < len(gripper_segments):
                        # Next segment exists, use its start as reference
                        next_seg_start, _ = gripper_segments[segment_idx + 1]
                        ref_frame_idx = next_seg_start
                    else:
                        # This is the last segment for this gripper
                        # Use end of episode as reference
                        ref_frame_idx = len(picker_positions) - 1
            
            # Get 3D position at reference frame
            ref_frame_idx = max(0, min(ref_frame_idx, len(picker_positions) - 1))
            ref_pos_3d = picker_positions[ref_frame_idx, gripper_id, :]
            ref_positions.append(ref_pos_3d)
        
        frame_to_ref_positions[frame_idx] = ref_positions
    
    # Create frames for GIF
    frames = []
    num_frames = len(images)
    img_h, img_w = images.shape[1], images.shape[2]
    
    # Camera parameters (tuned values)
    cam_pos = np.array([0.0, 0.3620, 0.0])  # x, y, z
    cam_angle = np.array([0.0, -90.0, 0.0])  # x, y, z rotations in degrees
    fov = 90.0  # degrees
    
    # Camera projection functions (copied from tune_camera_projection.py)
    def get_rotation_matrix(angle, axis):
        """Get rotation matrix around an axis."""
        axis = np.array(axis) / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        ux, uy, uz = axis
        m = np.zeros((4, 4))
        m[0][0] = cos_a + ux * ux * (1 - cos_a)
        m[0][1] = ux * uy * (1 - cos_a) - uz * sin_a
        m[0][2] = ux * uz * (1 - cos_a) + uy * sin_a
        m[1][0] = uy * ux * (1 - cos_a) + uz * sin_a
        m[1][1] = cos_a + uy * uy * (1 - cos_a)
        m[1][2] = uy * uz * (1 - cos_a) - ux * sin_a
        m[2][0] = uz * ux * (1 - cos_a) - uy * sin_a
        m[2][1] = uz * uy * (1 - cos_a) + ux * sin_a
        m[2][2] = cos_a + uz * uz * (1 - cos_a)
        m[3][3] = 1.0
        return m
    
    def intrinsic_from_fov(height, width, fov=90):
        """Get camera intrinsic matrix from FOV."""
        px, py = (width / 2, height / 2)
        hfov = fov / 360. * 2. * np.pi
        fx = width / (2. * np.tan(hfov / 2.))
        vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
        fy = height / (2. * np.tan(vfov / 2.))
        return np.array([[fx, 0, px, 0.],
                         [0, fy, py, 0.],
                         [0, 0, 1., 0.],
                         [0., 0., 0., 1.]])
    
    def world_to_pixel_projection(world_pos_3d, cam_pos, cam_angle, img_h, img_w, fov=90):
        """Project 3D world coordinates to 2D pixel coordinates using camera parameters."""
        K = intrinsic_from_fov(img_h, img_w, fov)
        cam_x_angle, cam_y_angle, cam_z_angle = cam_angle
        matrix1 = get_rotation_matrix(-cam_x_angle, [0, 1, 0])
        matrix2 = get_rotation_matrix(-cam_y_angle - np.pi, [1, 0, 0])
        rotation_matrix = matrix2 @ matrix1
        translation_matrix = np.eye(4)
        translation_matrix[0, 3] = -cam_pos[0]
        translation_matrix[1, 3] = -cam_pos[1]
        translation_matrix[2, 3] = -cam_pos[2]
        world_to_cam = rotation_matrix @ translation_matrix
        world_point = np.array([world_pos_3d[0], world_pos_3d[1], world_pos_3d[2], 1.0])
        cam_point = world_to_cam @ world_point
        if cam_point[2] <= 0:
            return None
        x_cam = cam_point[0] / cam_point[2]
        y_cam = cam_point[1] / cam_point[2]
        pixel_homogeneous = K @ np.array([x_cam, y_cam, 1.0, 1.0])
        x_pixel = pixel_homogeneous[0] / pixel_homogeneous[3]
        y_pixel = pixel_homogeneous[1] / pixel_homogeneous[3]
        if 0 <= x_pixel < img_w and 0 <= y_pixel < img_h:
            return (int(x_pixel), int(y_pixel))
        else:
            return None
    
    def world_to_pixel(world_pos_3d, img_w, img_h):
        """Convert 3D world coordinates to pixel coordinates using camera projection."""
        cam_angle_rad = np.radians(cam_angle)
        pixel = world_to_pixel_projection(
            world_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov
        )
        return pixel if pixel else None
    
    print(f"  Creating GIF with {num_frames} frames at {fps} fps...")
    for frame_idx in range(num_frames):
        # Create frame
        frame = images[frame_idx].copy()
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        # Convert to PIL for drawing
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil, 'RGBA')
        
        # Only draw reference positions for each gripper (no current positions, no trajectories)
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]  # Green, Blue, Magenta, Yellow
        ref_positions = frame_to_ref_positions[frame_idx]
        for gripper_id in range(num_pickers):
            ref_pos_3d = ref_positions[gripper_id]  # Already 3D position
            pixel = world_to_pixel(ref_pos_3d, img_w, img_h)
            
            if pixel is not None:
                x_ref, y_ref = pixel
                
                # Draw as a star (smaller marker)
                star_size = 8
                star_points = []
                for i in range(10):
                    angle = i * np.pi / 5 - np.pi / 2
                    r = star_size if i % 2 == 0 else star_size // 2
                    star_points.append((
                        x_ref + int(r * np.cos(angle)),
                        y_ref + int(r * np.sin(angle))
                    ))
                color = colors[gripper_id % len(colors)]
                draw.polygon(star_points, fill=(*color, 240), outline=(*color, 255), width=1)
        
        # Convert back to RGB (remove alpha channel)
        frame_pil = frame_pil.convert('RGB')
        frames.append(np.array(frame_pil))
    
    # Save as GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, frames, duration=1.0/fps, loop=0)
    print(f"Saved trajectory GIF with gripper references to: {output_path}")


def visualize_trajectory_gif(
    dataset_path: str,
    episode_idx: int,
    output_path: Path,
    fps: int = 10,
    include_images: bool = True,
):
    """Create a GIF showing the entire trajectory with reference frames.
    
    Args:
        dataset_path: Path to zarr dataset
        episode_idx: Index of episode to visualize
        output_path: Path to save GIF
        fps: Frames per second for the GIF
        include_images: Whether to use images from dataset (if available)
    """
    # Load dataset
    root = zarr.open(dataset_path, 'r')
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # Get episode boundaries
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    # Extract episode data
    states = data['state'][start_idx:end_idx]
    actions = data['action'][start_idx:end_idx]
    images = None
    if include_images and 'image' in data:
        images = data['image'][start_idx:end_idx]
    
    # Get metadata
    agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
    
    # Extract agent positions (first 2 dims of agent state)
    agent_positions = states[:, :2]
    
    # Create dataset to get reference frames
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        demo_indices=[episode_idx],
        use_contact_segmentation=True,
        contact_threshold=0.1,
        min_segment_length=5,
    )
    
    # Get segments for this episode
    demo_id = f"demo_{0}"  # Since we only loaded one episode, it's demo_0
    segments = dataset.segment_bounds.get(demo_id, [])
    
    # Build mapping from frame index to reference position
    # We need to find which segment each frame belongs to and get its reference
    from SD_pusht.utils.normalization import unnormalize_data
    
    frame_to_ref_pos = {}
    for frame_idx in range(len(agent_positions)):
        # Find which segment this frame belongs to
        ref_frame_idx = None
        for seg_start, seg_end in segments:
            if seg_start <= frame_idx < seg_end:
                # This frame is in this segment
                # Reference is the first frame of next segment (or end of episode)
                seg_idx = segments.index((seg_start, seg_end))
                if seg_idx + 1 < len(segments):
                    next_seg_start, _ = segments[seg_idx + 1]
                    ref_frame_idx = next_seg_start  # Relative to episode start
                else:
                    # Last segment, use end of episode
                    ref_frame_idx = len(agent_positions) - 1
                break
        
        if ref_frame_idx is None:
            # Frame not in any segment, use end of episode
            ref_frame_idx = len(agent_positions) - 1
        
        # Get reference position from the actual state data (unnormalized)
        # The reference position is the agent position at the reference frame
        ref_pos_unnorm = agent_positions[ref_frame_idx]
        frame_to_ref_pos[frame_idx] = ref_pos_unnorm
    
    # Create frames for GIF
    frames = []
    num_frames = len(agent_positions)
    
    # Get image dimensions
    if images is not None:
        img_h, img_w = images.shape[1], images.shape[2]
    else:
        # Create a blank canvas
        img_h, img_w = 256, 256
    
    # Normalize positions for visualization (assuming they're in some range)
    # We'll use the min/max of the actual positions
    pos_min = agent_positions.min(axis=0)
    pos_max = agent_positions.max(axis=0)
    pos_range = pos_max - pos_min
    pos_range = np.where(pos_range < 1e-6, 1.0, pos_range)  # Avoid division by zero
    
    def world_to_pixel(world_pos, img_w, img_h, margin=0.1):
        """Convert world coordinates to pixel coordinates with margin."""
        # Normalize to [0, 1] with margin
        x_norm = (world_pos[0] - pos_min[0]) / pos_range[0]
        y_norm = (world_pos[1] - pos_min[1]) / pos_range[1]
        # Add margin
        x_norm = margin + x_norm * (1 - 2 * margin)
        y_norm = margin + y_norm * (1 - 2 * margin)
        # Convert to pixel coordinates (flip y for image coordinates)
        x_pixel = int(x_norm * img_w)
        y_pixel = int((1 - y_norm) * img_h)  # Flip y
        return x_pixel, y_pixel
    
    print(f"  Creating GIF with {num_frames} frames...")
    for frame_idx in range(num_frames):
        # Create frame
        if images is not None:
            frame = images[frame_idx].copy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        else:
            # Create blank frame
            frame = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        
        # Convert to PIL for drawing
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil, 'RGBA')  # Use RGBA mode for transparency
        
        # Draw trajectory up to current frame
        trajectory_so_far = agent_positions[:frame_idx + 1]
        if len(trajectory_so_far) > 1:
            for i in range(len(trajectory_so_far) - 1):
                x1, y1 = world_to_pixel(trajectory_so_far[i], img_w, img_h)
                x2, y2 = world_to_pixel(trajectory_so_far[i + 1], img_w, img_h)
                # Draw line with increasing opacity (newer segments more visible)
                alpha = int(128 + 127 * (i / max(len(trajectory_so_far) - 1, 1)))
                draw.line([(x1, y1), (x2, y2)], fill=(100, 150, 255, alpha), width=3)
        
        # Draw current agent position (green circle)
        curr_pos = agent_positions[frame_idx]
        x_curr, y_curr = world_to_pixel(curr_pos, img_w, img_h)
        draw.ellipse([x_curr - 10, y_curr - 10, x_curr + 10, y_curr + 10], 
                    fill=(0, 255, 0, 255), outline=(0, 200, 0, 255), width=3)
        
        # Draw reference position for this frame (red star)
        ref_pos = frame_to_ref_pos[frame_idx]
        x_ref, y_ref = world_to_pixel(ref_pos, img_w, img_h)
        # Draw as a star/marker
        star_size = 15
        star_points = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2  # Start from top
            r = star_size if i % 2 == 0 else star_size // 2
            star_points.append((
                x_ref + int(r * np.cos(angle)),
                y_ref + int(r * np.sin(angle))
            ))
        draw.polygon(star_points, fill=(255, 0, 0, 255), outline=(200, 0, 0, 255), width=2)
        
        # Draw line from current position to reference position
        draw.line([(x_curr, y_curr), (x_ref, y_ref)], fill=(255, 255, 0, 180), width=2)
        
        # Draw segment boundaries if at segment start
        for seg_start, seg_end in segments:
            if frame_idx == seg_start:
                # Draw segment start marker (cyan square)
                seg_pos = agent_positions[seg_start]
                x_seg, y_seg = world_to_pixel(seg_pos, img_w, img_h)
                draw.rectangle([x_seg - 12, y_seg - 12, x_seg + 12, y_seg + 12],
                             fill=(0, 255, 255, 200), outline=(0, 200, 200, 255), width=2)
        
        # Add text overlay
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except:
                font = ImageFont.load_default()
        
        # Convert back to RGB (remove alpha channel)
        frame_pil = frame_pil.convert('RGB')
        frame_np = np.array(frame_pil)
        frames.append(frame_np)
    
    # Save as GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, frames, duration=1.0/fps, loop=0)
    print(f"Saved trajectory GIF to: {output_path}")


def visualize_batch_samples(
    dataset_path: str,
    num_samples: int,
    output_path: Path,
):
    """Visualize random samples from the dataset.
    
    Args:
        dataset_path: Path to zarr dataset
        num_samples: Number of samples to visualize
        output_path: Path to save visualization
    """
    # Create dataset
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        use_contact_segmentation=True,
        contact_threshold=0.1,
        min_segment_length=5,
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Get random samples
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        samples.append(batch)
    
    # Create visualization
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (sample, ax) in enumerate(zip(samples, axes)):
        obs = sample['obs'][0].numpy()  # (obs_horizon, obs_dim)
        action = sample['action'][0].numpy()  # (pred_horizon, action_dim)
        ref_pos = sample['reference_pos'][0].numpy()  # (2,)
        
        # Plot observation sequence (agent positions)
        obs_positions = obs[:, :2]  # First 2 dims are agent position
        ax.plot(obs_positions[:, 0], obs_positions[:, 1], 'bo-', 
               linewidth=2, markersize=6, label='Observations')
        
        # Plot reference position
        ax.scatter([ref_pos[0]], [ref_pos[1]], color='red', s=200, 
                  marker='*', label='Reference Position', zorder=5)
        
        # Plot action sequence (predicted positions)
        # Note: actions are normalized, so we need to denormalize for visualization
        # For now, just show the action values
        action_positions = action[:, :2] if action.shape[1] >= 2 else action
        ax.plot(action_positions[:, 0], action_positions[:, 1], 'g--', 
               linewidth=1, alpha=0.5, label='Actions (predicted)')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Sample {idx + 1}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Random Dataset Samples (n={num_samples})', fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved batch samples visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SoftGym zarr dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to zarr dataset")
    parser.add_argument("--output-dir", type=str, default="visualizations/softgym",
                       help="Output directory for visualizations")
    parser.add_argument("--num-episodes", type=int, default=5,
                       help="Number of episodes to visualize")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of random samples to visualize")
    parser.add_argument("--include-images", action="store_true",
                       help="Include images in episode visualizations (if available)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset info
    root = zarr.open(args.dataset_path, 'r')
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    num_episodes = len(episode_ends)
    
    print(f"Dataset: {args.dataset_path}")
    print(f"Total episodes: {num_episodes}")
    print(f"Total frames: {episode_ends[-1] if len(episode_ends) > 0 else 0}")
    print(f"Agent state dim: {meta.attrs.get('agent_state_dim', 'N/A')}")
    print(f"Env state dim: {meta.attrs.get('env_state_dim', 'N/A')}")
    print(f"Action dim: {meta.attrs.get('action_dim', 'N/A')}")
    print(f"Has images: {'image' in root['data']}")
    print()
    
    # Visualize episodes
    num_vis_episodes = min(args.num_episodes, num_episodes)
    print(f"Visualizing {num_vis_episodes} episodes...")
    for ep_idx in range(num_vis_episodes):
        print(f"  Episode {ep_idx}...")
        visualize_episode(
            args.dataset_path,
            ep_idx,
            output_dir / f"episode_{ep_idx}_trajectory.png",
            include_images=args.include_images,
        )
        visualize_segmentation(
            args.dataset_path,
            ep_idx,
            output_dir / f"episode_{ep_idx}_segmentation.png",
        )
        # Create trajectory GIF with reference frames (legacy)
        print(f"    Creating trajectory GIF...")
        visualize_trajectory_gif(
            args.dataset_path,
            ep_idx,
            output_dir / f"episode_{ep_idx}_trajectory.gif",
            fps=10,
            include_images=args.include_images,
        )
        # Create 2D plot to verify reference frames
        print(f"    Creating 2D reference frames plot...")
        visualize_gripper_refs_2d(
            args.dataset_path,
            ep_idx,
            output_dir / f"episode_{ep_idx}_gripper_refs_2d.png",
        )
        
        # Create trajectory GIF with gripper reference frames projected on images
        if args.include_images:
            print(f"    Creating trajectory GIF with gripper references...")
            visualize_trajectory_gif_with_gripper_refs(
                args.dataset_path,
                ep_idx,
                output_dir / f"episode_{ep_idx}_gripper_refs.gif",
                fps=0.5,  # Very slow (2 seconds per frame) for better viewing
                include_images=True,
            )
    
    # Visualize random samples
    print(f"\nVisualizing {args.num_samples} random samples...")
    visualize_batch_samples(
        args.dataset_path,
        args.num_samples,
        output_dir / "batch_samples.png",
    )
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
