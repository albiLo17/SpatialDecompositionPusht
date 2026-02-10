#!/usr/bin/env python3
"""CLI tool to tune camera projection parameters for 3D to 2D projection.

This tool allows you to:
- Load a dataset and select an episode/frame
- Select which gripper to visualize
- Specify camera parameters via command-line arguments
- Save an image with the projected gripper position

Example:
    python tune_camera_projection.py \
        --dataset-path datasets/ropeflatten_expert.zarr \
        --episode-idx 0 \
        --frame-idx 0 \
        --gripper-id 0 \
        --cam-x 0.0 --cam-y 0.85 --cam-z 0.0 \
        --rot-x 0.0 --rot-y -90.0 --rot-z 0.0 \
        --fov 90.0 \
        --output output.png
"""

import argparse
import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path


def get_rotation_matrix(angle, axis):
    """Get rotation matrix around an axis.
    
    Args:
        angle: Rotation angle in radians
        axis: Rotation axis (3D vector)
    
    Returns:
        4x4 rotation matrix
    """
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
    """Get camera intrinsic matrix from FOV.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        fov: Field of view in degrees
    
    Returns:
        4x4 intrinsic matrix
    """
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
    """Project 3D world coordinates to 2D pixel coordinates using camera parameters.
    
    Args:
        world_pos_3d: 3D position in world coordinates (x, y, z)
        cam_pos: Camera position (x, y, z)
        cam_angle: Camera rotation angles in radians (x, y, z)
        img_h: Image height in pixels
        img_w: Image width in pixels
        fov: Field of view in degrees
    
    Returns:
        (x_pixel, y_pixel) or None if point is behind camera
    """
    # Get intrinsic matrix
    K = intrinsic_from_fov(img_h, img_w, fov)
    
    # Get rotation matrix: from world to camera
    cam_x_angle, cam_y_angle, cam_z_angle = cam_angle
    matrix1 = get_rotation_matrix(-cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(-cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1
    
    # Get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0, 3] = -cam_pos[0]
    translation_matrix[1, 3] = -cam_pos[1]
    translation_matrix[2, 3] = -cam_pos[2]
    
    # World to camera transformation
    world_to_cam = rotation_matrix @ translation_matrix
    
    # Transform world point to camera coordinates
    world_point = np.array([world_pos_3d[0], world_pos_3d[1], world_pos_3d[2], 1.0])
    cam_point = world_to_cam @ world_point
    
    # Check if point is behind camera (z < 0 in camera space)
    if cam_point[2] <= 0:
        return None
    
    # Project to image plane
    x_cam = cam_point[0] / cam_point[2]
    y_cam = cam_point[1] / cam_point[2]
    
    # Apply intrinsics
    pixel_homogeneous = K @ np.array([x_cam, y_cam, 1.0, 1.0])
    x_pixel = pixel_homogeneous[0] / pixel_homogeneous[3]
    y_pixel = pixel_homogeneous[1] / pixel_homogeneous[3]
    
    # Check if point is within image bounds
    if 0 <= x_pixel < img_w and 0 <= y_pixel < img_h:
        return (int(x_pixel), int(y_pixel))
    else:
        return None


def create_projection_image(dataset_path, episode_idx, frame_idx, gripper_id,
                           cam_pos, cam_angle, fov, output_path):
    """Create and save an image with projected gripper position.
    
    Args:
        dataset_path: Path to zarr dataset
        episode_idx: Episode index to load
        frame_idx: Frame index within episode
        gripper_id: Gripper ID to visualize (0 or 1)
        cam_pos: Camera position (x, y, z)
        cam_angle: Camera rotation angles in degrees (x, y, z)
        fov: Field of view in degrees
        output_path: Path to save the output image
    """
    # Load dataset
    root = zarr.open(dataset_path, 'r')
    data = root['data']
    meta = root['meta']
    episode_ends = meta['episode_ends'][:]
    
    # Get episode boundaries
    start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
    end_idx = episode_ends[episode_idx]
    
    # Load episode data
    states = data['state'][start_idx:end_idx]
    images = None
    if 'image' in data:
        images = data['image'][start_idx:end_idx]
    
    # Get metadata
    agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
    num_pickers = int(meta.attrs.get('num_picker', 2))
    
    # Extract picker positions
    picker_positions = states[:, :agent_state_dim].reshape(
        len(states), num_pickers, 3
    )
    gripper_positions = picker_positions[:, gripper_id, :]  # (num_frames, 3)
    
    # Get current frame data
    current_image = None
    if images is not None:
        current_image = images[frame_idx].copy()
        if current_image.dtype != np.uint8:
            if current_image.max() <= 1.0:
                current_image = (current_image * 255).astype(np.uint8)
            else:
                current_image = current_image.astype(np.uint8)
    
    current_gripper_pos = gripper_positions[frame_idx]  # (3,)
    
    # Get image dimensions
    if current_image is not None:
        img_h, img_w = current_image.shape[:2]
    else:
        img_h, img_w = 128, 128
    
    # Convert camera angles from degrees to radians
    cam_angle_rad = np.radians(cam_angle)
    
    # Project gripper position
    pixel = world_to_pixel_projection(
        current_gripper_pos, cam_pos, cam_angle_rad,
        img_h, img_w, fov
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Display image
    if current_image is not None:
        ax.imshow(current_image)
    else:
        # Create blank image
        blank = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        ax.imshow(blank)
    
    # Draw projected gripper position
    if pixel:
        # Draw gripper position as a circle (smaller marker)
        circle = Circle(pixel, 5, color='red', fill=True, alpha=0.7, linewidth=1)
        ax.add_patch(circle)
        # Draw crosshair (smaller)
        ax.plot([pixel[0] - 8, pixel[0] + 8], [pixel[1], pixel[1]], 'r-', linewidth=1)
        ax.plot([pixel[0], pixel[0]], [pixel[1] - 8, pixel[1] + 8], 'r-', linewidth=1)
    
    # Add text overlay
    info_text = (
        f"Episode {episode_idx} | Frame {frame_idx} | Gripper {gripper_id}\n"
        f"World: [{current_gripper_pos[0]:.3f}, {current_gripper_pos[1]:.3f}, {current_gripper_pos[2]:.3f}]\n"
        f"Cam: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]\n"
        f"Rot: [{cam_angle[0]:.1f}°, {cam_angle[1]:.1f}°, {cam_angle[2]:.1f}°]\n"
        f"FOV: {fov:.1f}°"
    )
    if pixel:
        info_text += f"\nPixel: ({pixel[0]}, {pixel[1]})"
    else:
        info_text += "\nPixel: None (behind camera or out of bounds)"
    
    ax.text(10, 10, info_text, fontsize=12, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            verticalalignment='top', family='monospace')
    
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)  # Flip y-axis for image coordinates
    ax.set_title(f'Camera Projection - Episode {episode_idx} Frame {frame_idx} Gripper {gripper_id}', fontsize=14)
    ax.axis('off')
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print information
    print("="*60)
    print("Camera Projection Results")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Episode: {episode_idx}, Frame: {frame_idx}, Gripper: {gripper_id}")
    print(f"Gripper World Position: [{current_gripper_pos[0]:.4f}, {current_gripper_pos[1]:.4f}, {current_gripper_pos[2]:.4f}]")
    print(f"Camera Position: [{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")
    print(f"Camera Rotation (deg): [{cam_angle[0]:.2f}, {cam_angle[1]:.2f}, {cam_angle[2]:.2f}]")
    print(f"FOV: {fov:.2f} degrees")
    if pixel:
        print(f"Projected Pixel: ({pixel[0]}, {pixel[1]})")
    else:
        print("Projected Pixel: None (point is behind camera or out of bounds)")
    print(f"Output saved to: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to tune camera projection parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python tune_camera_projection.py \\
      --dataset-path datasets/ropeflatten_expert.zarr \\
      --episode-idx 0 --frame-idx 0 --gripper-id 0 \\
      --cam-x 0.0 --cam-y 0.85 --cam-z 0.0 \\
      --rot-x 0.0 --rot-y -90.0 --rot-z 0.0 \\
      --fov 90.0 \\
      --output projection_test.png
        """
    )
    
    # Dataset arguments
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to zarr dataset")
    parser.add_argument("--episode-idx", type=int, default=0,
                       help="Episode index to load")
    parser.add_argument("--frame-idx", type=int, default=0,
                       help="Frame index within episode")
    parser.add_argument("--gripper-id", type=int, default=0,
                       help="Gripper ID to visualize (0 or 1)")
    
    # Camera position arguments
    parser.add_argument("--cam-x", type=float, default=0.0,
                       help="Camera X position")
    parser.add_argument("--cam-y", type=float, default=0.3620,
                       help="Camera Y position (height)")
    parser.add_argument("--cam-z", type=float, default=0.0,
                       help="Camera Z position")
    
    # Camera rotation arguments (in degrees)
    parser.add_argument("--rot-x", type=float, default=0.0,
                       help="Camera X rotation in degrees")
    parser.add_argument("--rot-y", type=float, default=-90.0,
                       help="Camera Y rotation in degrees")
    parser.add_argument("--rot-z", type=float, default=0.0,
                       help="Camera Z rotation in degrees")
    
    # FOV argument
    parser.add_argument("--fov", type=float, default=90.0,
                       help="Field of view in degrees")
    
    # Output argument
    parser.add_argument("--output", type=str, default="camera_projection.png",
                       help="Output image path")
    
    args = parser.parse_args()
    
    # Prepare camera parameters
    cam_pos = np.array([args.cam_x, args.cam_y, args.cam_z])
    cam_angle = np.array([args.rot_x, args.rot_y, args.rot_z])
    
    # Create and save projection image
    create_projection_image(
        dataset_path=args.dataset_path,
        episode_idx=args.episode_idx,
        frame_idx=args.frame_idx,
        gripper_id=args.gripper_id,
        cam_pos=cam_pos,
        cam_angle=cam_angle,
        fov=args.fov,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
