"""SoftGym visualization utilities for evaluation."""

import os
import numpy as np
import torch
import zarr
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from SD_pusht.utils.normalization import normalize_data, unnormalize_data


# Camera parameters (tuned values)
DEFAULT_CAM_POS = np.array([0.0, 0.3620, 0.0])  # x, y, z
DEFAULT_CAM_ANGLE = np.array([0.0, -90.0, 0.0])  # x, y, z rotations in degrees
DEFAULT_FOV = 90.0  # degrees


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
    
    # Apply intrinsic matrix
    pixel_homogeneous = K @ np.array([x_cam, y_cam, 1.0, 1.0])
    x_pixel = pixel_homogeneous[0] / pixel_homogeneous[3]
    y_pixel = pixel_homogeneous[1] / pixel_homogeneous[3]
    
    # Check if within image bounds
    if 0 <= x_pixel < img_w and 0 <= y_pixel < img_h:
        return (int(x_pixel), int(y_pixel))
    else:
        return None


def extract_gripper_positions_3d(obs: np.ndarray, agent_state_dim: int, num_pickers: int) -> np.ndarray:
    """Extract 3D gripper positions from observation.
    
    Args:
        obs: Observation array, shape (obs_horizon, obs_dim) or (obs_dim,)
        agent_state_dim: Dimension of agent state (number of picker positions)
        num_pickers: Number of pickers/grippers
    
    Returns:
        Array of shape (num_pickers, 3) with (x, y, z) positions for each gripper
    """
    if obs.ndim == 2:
        # Use the last observation in the sequence
        obs_vec = obs[-1]
    else:
        obs_vec = obs
    
    # Agent state is at the end of observation
    # For SoftGym: agent state contains picker positions [x1, y1, z1, x2, y2, z2, ...]
    agent_state = obs_vec[-agent_state_dim:]
    
    gripper_positions = []
    for gripper_id in range(num_pickers):
        start_idx = gripper_id * 3
        gripper_pos = agent_state[start_idx:start_idx + 3]  # (x, y, z)
        gripper_positions.append(gripper_pos)
    
    return np.array(gripper_positions)  # (num_pickers, 3)


def load_image_from_dataset(dataset_path: str, sample_idx: int, episode_ends: np.ndarray, 
                            indices: np.ndarray) -> Optional[np.ndarray]:
    """Load image from zarr dataset for a given sample index.
    
    Args:
        dataset_path: Path to zarr dataset
        sample_idx: Sample index in the dataset
        episode_ends: Array of episode end indices (global)
        indices: Sample indices array from dataset
    
    Returns:
        Image array (H, W, C) as uint8, or None if not available
    """
    try:
        root = zarr.open(dataset_path, 'r')
        if 'image' not in root['data']:
            return None
        
        # Get the buffer_start_idx for this sample
        buffer_start_idx, _, _, _ = indices[sample_idx]
        
        # Find which episode this belongs to
        episode_idx = 0
        for i, ep_end in enumerate(episode_ends):
            if buffer_start_idx < ep_end:
                episode_idx = i
                break
        
        # Get episode boundaries
        start_idx = episode_ends[episode_idx - 1] if episode_idx > 0 else 0
        end_idx = episode_ends[episode_idx]
        
        # Get relative frame index within episode
        rel_frame_idx = buffer_start_idx - start_idx
        
        # Load image
        images = root['data']['image']
        if rel_frame_idx < len(images):
            img = images[start_idx + rel_frame_idx]
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            return img
    except Exception:
        pass
    
    return None


def visualize_action_predictions_softgym(
    model,
    dataset,
    stats,
    dataset_path: str,
    sample_idx=0,
    out_path="action_prediction_vis_softgym.png",
    device=None,
    cam_pos: Optional[np.ndarray] = None,
    cam_angle: Optional[np.ndarray] = None,
    fov: float = 90.0,
):
    """
    Visualize action policy predictions vs ground truth actions on a SoftGym image.
    
    Args:
        model: LocalFlowPolicy3D model
        dataset: SoftGymSegmentedDatasetSimple instance
        stats: Dataset statistics for unnormalization
        dataset_path: Path to zarr dataset (for loading images)
        sample_idx: Index of the sample to visualize
        out_path: Path to save the visualization
        device: Device to run model on
        cam_pos: Camera position (x, y, z). If None, uses default.
        cam_angle: Camera rotation in degrees (x, y, z). If None, uses default.
        fov: Field of view in degrees
    
    Returns:
        Dictionary with visualization information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    obs_seq = torch.from_numpy(sample['obs']).float().unsqueeze(0).to(device)  # (1, obs_horizon, obs_dim)
    gt_actions = torch.from_numpy(sample['action']).float().unsqueeze(0).to(device)  # (1, pred_horizon, act_dim)
    gt_ref_pos = torch.from_numpy(sample['reference_pos']).float().unsqueeze(0).to(device)  # (1, num_pickers, 3)
    
    # Get initial observation (for extracting gripper positions)
    initial_obs = sample['obs'][0]  # (obs_dim,)
    
    # Get reference position stats (3D picker positions)
    reference_pos_stats = None
    if stats and "obs" in stats:
        obs_dim = stats["obs"]["min"].shape[0]
        num_pickers = dataset.num_pickers
        picker_start_idx = dataset.agent_state_dim - 3 * num_pickers
        picker_x_indices = [picker_start_idx + i * 3 for i in range(num_pickers)]
        picker_y_indices = [picker_start_idx + i * 3 + 1 for i in range(num_pickers)]
        picker_z_indices = [picker_start_idx + i * 3 + 2 for i in range(num_pickers)]
        
        x_mins = [stats["obs"]['min'][idx] for idx in picker_x_indices]
        y_mins = [stats["obs"]['min'][idx] for idx in picker_y_indices]
        z_mins = [stats["obs"]['min'][idx] for idx in picker_z_indices]
        x_maxs = [stats["obs"]['max'][idx] for idx in picker_x_indices]
        y_maxs = [stats["obs"]['max'][idx] for idx in picker_y_indices]
        z_maxs = [stats["obs"]['max'][idx] for idx in picker_z_indices]
        
        reference_pos_stats = {
            'min': np.array([min(x_mins), min(y_mins), min(z_mins)]),
            'max': np.array([max(x_maxs), max(y_maxs), max(z_maxs)]),
        }
    
    # Get predictions from model
    action_stats = stats.get("action", None) if stats else None
    
    with torch.no_grad():
        predictions = model.get_action(
            obs_seq, 
            reference_position=gt_ref_pos,
            action_stats=action_stats,
            reference_pos_stats=reference_pos_stats,
        )
        pred_world_actions = predictions["actions"]  # (1, action_horizon, act_dim) - UNNORMALIZED world actions
        pred_ref_pos = predictions.get("reference_pos", gt_ref_pos)  # (1, num_pickers, 3) - normalized
    
    # Convert to numpy
    pred_world_actions_np = pred_world_actions.cpu().numpy()[0]  # (action_horizon, act_dim)
    gt_actions_np = gt_actions.cpu().numpy()[0]  # (pred_horizon, act_dim)
    pred_ref_pos_np = pred_ref_pos.cpu().numpy()[0]  # (num_pickers, 3)
    gt_ref_pos_np = gt_ref_pos.cpu().numpy()[0]  # (num_pickers, 3)
    
    # Unnormalize reference positions
    if reference_pos_stats is not None:
        gt_ref_pos_unnorm = unnormalize_data(gt_ref_pos_np[None, :, :], reference_pos_stats)[0]  # (num_pickers, 3)
        pred_ref_pos_unnorm = unnormalize_data(pred_ref_pos_np[None, :, :], reference_pos_stats)[0]  # (num_pickers, 3)
    else:
        gt_ref_pos_unnorm = gt_ref_pos_np
        pred_ref_pos_unnorm = pred_ref_pos_np
    
    # Extract initial gripper positions
    initial_gripper_positions = extract_gripper_positions_3d(
        initial_obs, dataset.agent_state_dim, dataset.num_pickers
    )
    
    # Unnormalize initial gripper positions
    if stats and "obs" in stats:
        obs_unnorm = unnormalize_data(initial_obs[None, :], stats["obs"])[0]
        initial_gripper_positions = extract_gripper_positions_3d(
            obs_unnorm, dataset.agent_state_dim, dataset.num_pickers
        )
    
    # Unnormalize actions
    if action_stats is not None:
        # Predicted actions are already unnormalized
        pred_actions_unnorm = pred_world_actions_np
        # Unnormalize GT actions
        gt_actions_unnorm = unnormalize_data(gt_actions_np[:len(pred_actions_unnorm)], action_stats)
    else:
        pred_actions_unnorm = pred_world_actions_np
        gt_actions_unnorm = gt_actions_np[:len(pred_actions_unnorm)]
    
    # Compute action error (L2 norm of translation components)
    action_error = 0.0
    num_valid = 0
    for t in range(min(len(pred_actions_unnorm), len(gt_actions_unnorm))):
        for gripper_id in range(dataset.num_pickers):
            action_start_idx = gripper_id * 4
            pred_trans = pred_actions_unnorm[t, action_start_idx:action_start_idx + 3]
            gt_trans = gt_actions_unnorm[t, action_start_idx:action_start_idx + 3]
            action_error += np.linalg.norm(pred_trans - gt_trans)
            num_valid += 1
    if num_valid > 0:
        action_error /= num_valid
    
    # Load image from dataset
    image = load_image_from_dataset(dataset_path, sample_idx, dataset.episode_ends, dataset.indices)
    
    # Set up camera parameters
    if cam_pos is None:
        cam_pos = DEFAULT_CAM_POS
    if cam_angle is None:
        cam_angle = DEFAULT_CAM_ANGLE
    
    cam_angle_rad = np.radians(cam_angle)
    
    # Get image dimensions
    if image is not None:
        img_h, img_w = image.shape[:2]
        frame = image.copy()
    else:
        # Create blank image
        img_h, img_w = 128, 128
        frame = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
    
    # Colors for each gripper
    gripper_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]  # Yellow, Magenta, Cyan, Green
    
    # Draw initial gripper positions
    for gripper_id in range(dataset.num_pickers):
        gripper_pos_3d = initial_gripper_positions[gripper_id]
        pixel = world_to_pixel_projection(gripper_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
        if pixel is not None:
            color = gripper_colors[gripper_id % len(gripper_colors)]
            cv2.circle(frame_bgr, pixel, 8, color, -1)
            cv2.circle(frame_bgr, pixel, 8, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"G{gripper_id}",
                       (pixel[0] + 10, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw reference positions
    for gripper_id in range(dataset.num_pickers):
        # GT reference position
        ref_pos_3d = gt_ref_pos_unnorm[gripper_id]
        pixel = world_to_pixel_projection(ref_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
        if pixel is not None:
            color = gripper_colors[gripper_id % len(gripper_colors)]
            # Draw as star
            star_size = 10
            star_points = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi / 2
                r = star_size if i % 2 == 0 else star_size // 2
                star_points.append((
                    pixel[0] + int(r * np.cos(angle)),
                    pixel[1] + int(r * np.sin(angle))
                ))
            pts = np.array(star_points, np.int32)
            cv2.fillPoly(frame_bgr, [pts], color)
            cv2.polylines(frame_bgr, [pts], True, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f"GT{gripper_id}",
                       (pixel[0] + 15, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw predicted action sequences (for each gripper)
    action_horizon = len(pred_actions_unnorm)
    for gripper_id in range(dataset.num_pickers):
        action_start_idx = gripper_id * 4
        color = gripper_colors[gripper_id % len(gripper_colors)]
        
        # Draw predicted action sequence
        prev_pos_3d = initial_gripper_positions[gripper_id].copy()
        for t in range(action_horizon):
            # Get translation from action
            dx, dy, dz = pred_actions_unnorm[t, action_start_idx:action_start_idx + 3]
            # Compute next position (in world frame, actions are already in world frame)
            next_pos_3d = prev_pos_3d + np.array([dx, dy, dz])
            
            # Project to image
            pixel = world_to_pixel_projection(next_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
            if pixel is not None:
                cv2.circle(frame_bgr, pixel, 5, color, -1)
                if t > 0:
                    prev_pixel = world_to_pixel_projection(prev_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
                    if prev_pixel is not None:
                        cv2.line(frame_bgr, prev_pixel, pixel, color, 2)
                cv2.putText(frame_bgr, f"P{t}",
                           (pixel[0] + 8, pixel[1] + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            prev_pos_3d = next_pos_3d
    
    # Draw GT action sequences
    for gripper_id in range(dataset.num_pickers):
        action_start_idx = gripper_id * 4
        color = (0, 255, 0)  # Green for GT
        
        prev_pos_3d = initial_gripper_positions[gripper_id].copy()
        for t in range(min(action_horizon, len(gt_actions_unnorm))):
            dx, dy, dz = gt_actions_unnorm[t, action_start_idx:action_start_idx + 3]
            next_pos_3d = prev_pos_3d + np.array([dx, dy, dz])
            
            pixel = world_to_pixel_projection(next_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
            if pixel is not None:
                cv2.circle(frame_bgr, pixel, 5, color, -1)
                if t > 0:
                    prev_pixel = world_to_pixel_projection(prev_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
                    if prev_pixel is not None:
                        cv2.line(frame_bgr, prev_pixel, pixel, color, 2)
                cv2.putText(frame_bgr, f"GT{t}",
                           (pixel[0] + 8, pixel[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            prev_pos_3d = next_pos_3d
    
    # Add text with error
    error_text = f"Action Error: {action_error:.4f}"
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Convert back to RGB for saving
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.imsave(out_path, frame_rgb)
    
    return {
        "image_path": out_path,
        "gt_actions": gt_actions_unnorm,
        "pred_actions": pred_actions_unnorm,
        "action_error": action_error,
        "initial_gripper_positions": initial_gripper_positions,
    }


def visualize_position_predictions_softgym(
    model,
    dataset,
    stats,
    dataset_path: str,
    sample_idx=0,
    out_path="position_prediction_vis_softgym.png",
    device=None,
    cam_pos: Optional[np.ndarray] = None,
    cam_angle: Optional[np.ndarray] = None,
    fov: float = 90.0,
):
    """
    Visualize position decoder predictions vs ground truth reference positions on a SoftGym image.
    
    Args:
        model: LocalFlowPolicy3D model with position decoder
        dataset: SoftGymSegmentedDatasetSimple instance
        stats: Dataset statistics for unnormalization
        dataset_path: Path to zarr dataset (for loading images)
        sample_idx: Index of the sample to visualize
        out_path: Path to save the visualization
        device: Device to run model on
        cam_pos: Camera position (x, y, z). If None, uses default.
        cam_angle: Camera rotation in degrees (x, y, z). If None, uses default.
        fov: Field of view in degrees
    
    Returns:
        Dictionary with visualization information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.position_decoder is None:
        raise ValueError("Model does not have a position decoder")
    
    model.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    obs_seq = torch.from_numpy(sample['obs']).float().unsqueeze(0).to(device)  # (1, obs_horizon, obs_dim)
    gt_ref_pos = torch.from_numpy(sample['reference_pos']).float().unsqueeze(0).to(device)  # (1, num_pickers, 3)
    
    # Get initial observation
    initial_obs = sample['obs'][0]  # (obs_dim,)
    
    # Get reference position stats
    reference_pos_stats = None
    if stats and "obs" in stats:
        obs_dim = stats["obs"]["min"].shape[0]
        num_pickers = dataset.num_pickers
        picker_start_idx = dataset.agent_state_dim - 3 * num_pickers
        picker_x_indices = [picker_start_idx + i * 3 for i in range(num_pickers)]
        picker_y_indices = [picker_start_idx + i * 3 + 1 for i in range(num_pickers)]
        picker_z_indices = [picker_start_idx + i * 3 + 2 for i in range(num_pickers)]
        
        x_mins = [stats["obs"]['min'][idx] for idx in picker_x_indices]
        y_mins = [stats["obs"]['min'][idx] for idx in picker_y_indices]
        z_mins = [stats["obs"]['min'][idx] for idx in picker_z_indices]
        x_maxs = [stats["obs"]['max'][idx] for idx in picker_x_indices]
        y_maxs = [stats["obs"]['max'][idx] for idx in picker_y_indices]
        z_maxs = [stats["obs"]['max'][idx] for idx in picker_z_indices]
        
        reference_pos_stats = {
            'min': np.array([min(x_mins), min(y_mins), min(z_mins)]),
            'max': np.array([max(x_maxs), max(y_maxs), max(z_maxs)]),
        }
    
    # Get predictions from model
    with torch.no_grad():
        # Encode observations
        obs_flat = obs_seq.flatten(start_dim=1)  # (1, obs_horizon * obs_dim)
        obs_encoded = model.obs_encoder(obs_flat)  # (1, obs_encoder_dim)
        
        # Predict reference positions
        pred_ref_pos = model.position_decoder(obs_encoded, x_init=None)  # (1, num_pickers, 3)
    
    # Convert to numpy
    pred_ref_pos_np = pred_ref_pos.cpu().numpy()[0]  # (num_pickers, 3)
    gt_ref_pos_np = gt_ref_pos.cpu().numpy()[0]  # (num_pickers, 3)
    
    # Unnormalize reference positions
    if reference_pos_stats is not None:
        gt_ref_pos_unnorm = unnormalize_data(gt_ref_pos_np[None, :, :], reference_pos_stats)[0]  # (num_pickers, 3)
        pred_ref_pos_unnorm = unnormalize_data(pred_ref_pos_np[None, :, :], reference_pos_stats)[0]  # (num_pickers, 3)
    else:
        gt_ref_pos_unnorm = gt_ref_pos_np
        pred_ref_pos_unnorm = pred_ref_pos_np
    
    # Extract initial gripper positions
    initial_gripper_positions = extract_gripper_positions_3d(
        initial_obs, dataset.agent_state_dim, dataset.num_pickers
    )
    
    # Unnormalize initial gripper positions
    if stats and "obs" in stats:
        obs_unnorm = unnormalize_data(initial_obs[None, :], stats["obs"])[0]
        initial_gripper_positions = extract_gripper_positions_3d(
            obs_unnorm, dataset.agent_state_dim, dataset.num_pickers
        )
    
    # Compute prediction error
    pred_error = np.mean(np.linalg.norm(pred_ref_pos_unnorm - gt_ref_pos_unnorm, axis=1))
    
    # Load image from dataset
    image = load_image_from_dataset(dataset_path, sample_idx, dataset.episode_ends, dataset.indices)
    
    # Set up camera parameters
    if cam_pos is None:
        cam_pos = DEFAULT_CAM_POS
    if cam_angle is None:
        cam_angle = DEFAULT_CAM_ANGLE
    
    cam_angle_rad = np.radians(cam_angle)
    
    # Get image dimensions
    if image is not None:
        img_h, img_w = image.shape[:2]
        frame = image.copy()
    else:
        img_h, img_w = 128, 128
        frame = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
    
    # Colors for each gripper
    gripper_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]  # Yellow, Magenta, Cyan, Green
    
    # Draw initial gripper positions
    for gripper_id in range(dataset.num_pickers):
        gripper_pos_3d = initial_gripper_positions[gripper_id]
        pixel = world_to_pixel_projection(gripper_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
        if pixel is not None:
            color = gripper_colors[gripper_id % len(gripper_colors)]
            cv2.circle(frame_bgr, pixel, 8, color, -1)
            cv2.circle(frame_bgr, pixel, 8, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"G{gripper_id}",
                       (pixel[0] + 10, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw GT reference positions (green stars)
    for gripper_id in range(dataset.num_pickers):
        ref_pos_3d = gt_ref_pos_unnorm[gripper_id]
        pixel = world_to_pixel_projection(ref_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
        if pixel is not None:
            color = (0, 255, 0)  # Green for GT
            star_size = 12
            star_points = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi / 2
                r = star_size if i % 2 == 0 else star_size // 2
                star_points.append((
                    pixel[0] + int(r * np.cos(angle)),
                    pixel[1] + int(r * np.sin(angle))
                ))
            pts = np.array(star_points, np.int32)
            cv2.fillPoly(frame_bgr, [pts], color)
            cv2.polylines(frame_bgr, [pts], True, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"GT{gripper_id}",
                       (pixel[0] + 15, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw predicted reference positions (red stars)
    for gripper_id in range(dataset.num_pickers):
        ref_pos_3d = pred_ref_pos_unnorm[gripper_id]
        pixel = world_to_pixel_projection(ref_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
        if pixel is not None:
            color = (0, 0, 255)  # Red for prediction
            star_size = 12
            star_points = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi / 2
                r = star_size if i % 2 == 0 else star_size // 2
                star_points.append((
                    pixel[0] + int(r * np.cos(angle)),
                    pixel[1] + int(r * np.sin(angle))
                ))
            pts = np.array(star_points, np.int32)
            cv2.fillPoly(frame_bgr, [pts], color)
            cv2.polylines(frame_bgr, [pts], True, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Pred{gripper_id}",
                       (pixel[0] + 15, pixel[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw line from GT to prediction
            gt_pixel = world_to_pixel_projection(gt_ref_pos_unnorm[gripper_id], cam_pos, cam_angle_rad, img_h, img_w, fov)
            if gt_pixel is not None:
                cv2.line(frame_bgr, gt_pixel, pixel, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Add text with error
    error_text = f"Position Error: {pred_error:.4f}"
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Convert back to RGB for saving
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.imsave(out_path, frame_rgb)
    
    return {
        "image_path": out_path,
        "gt_ref_pos": gt_ref_pos_unnorm,
        "pred_ref_pos": pred_ref_pos_unnorm,
        "pred_error": pred_error,
        "initial_gripper_positions": initial_gripper_positions,
    }


def visualize_training_trajectory_softgym(
    model,
    dataset,
    stats,
    dataset_path: str,
    sample_idx=0,
    out_path="trajectory_vis_softgym.png",
    device=None,
    cam_pos: Optional[np.ndarray] = None,
    cam_angle: Optional[np.ndarray] = None,
    fov: float = 90.0,
):
    """
    Visualize full predicted trajectory on a SoftGym image.
    
    Args:
        model: LocalFlowPolicy3D model
        dataset: SoftGymSegmentedDatasetSimple instance
        stats: Dataset statistics for unnormalization
        dataset_path: Path to zarr dataset (for loading images)
        sample_idx: Index of the sample to visualize
        out_path: Path to save the visualization
        device: Device to run model on
        cam_pos: Camera position (x, y, z). If None, uses default.
        cam_angle: Camera rotation in degrees (x, y, z). If None, uses default.
        fov: Field of view in degrees
    
    Returns:
        Dictionary with visualization information
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    obs_seq = torch.from_numpy(sample['obs']).float().unsqueeze(0).to(device)  # (1, obs_horizon, obs_dim)
    gt_ref_pos = torch.from_numpy(sample['reference_pos']).float().unsqueeze(0).to(device)  # (1, num_pickers, 3)
    
    # Get initial observation
    initial_obs = sample['obs'][0]  # (obs_dim,)
    
    # Get reference position stats
    reference_pos_stats = None
    if stats and "obs" in stats:
        obs_dim = stats["obs"]["min"].shape[0]
        num_pickers = dataset.num_pickers
        picker_start_idx = dataset.agent_state_dim - 3 * num_pickers
        picker_x_indices = [picker_start_idx + i * 3 for i in range(num_pickers)]
        picker_y_indices = [picker_start_idx + i * 3 + 1 for i in range(num_pickers)]
        picker_z_indices = [picker_start_idx + i * 3 + 2 for i in range(num_pickers)]
        
        x_mins = [stats["obs"]['min'][idx] for idx in picker_x_indices]
        y_mins = [stats["obs"]['min'][idx] for idx in picker_y_indices]
        z_mins = [stats["obs"]['min'][idx] for idx in picker_z_indices]
        x_maxs = [stats["obs"]['max'][idx] for idx in picker_x_indices]
        y_maxs = [stats["obs"]['max'][idx] for idx in picker_y_indices]
        z_maxs = [stats["obs"]['max'][idx] for idx in picker_z_indices]
        
        reference_pos_stats = {
            'min': np.array([min(x_mins), min(y_mins), min(z_mins)]),
            'max': np.array([max(x_maxs), max(y_maxs), max(z_maxs)]),
        }
    
    # Get predictions from model
    action_stats = stats.get("action", None) if stats else None
    
    with torch.no_grad():
        predictions = model.get_action(
            obs_seq,
            reference_position=gt_ref_pos,
            action_stats=action_stats,
            reference_pos_stats=reference_pos_stats,
        )
        pred_ref_pos = predictions.get("reference_pos", gt_ref_pos)  # (1, num_pickers, 3) - normalized
        pred_actions = predictions["actions"]  # (1, action_horizon, act_dim) - UNNORMALIZED world actions
    
    # Convert to numpy
    pred_ref_pos_np = pred_ref_pos.cpu().numpy()[0]  # (num_pickers, 3)
    pred_actions_np = pred_actions.cpu().numpy()[0]  # (action_horizon, act_dim)
    
    # Unnormalize reference positions
    if reference_pos_stats is not None:
        pred_ref_pos_unnorm = unnormalize_data(pred_ref_pos_np[None, :, :], reference_pos_stats)[0]  # (num_pickers, 3)
    else:
        pred_ref_pos_unnorm = pred_ref_pos_np
    
    # Extract initial gripper positions
    initial_gripper_positions = extract_gripper_positions_3d(
        initial_obs, dataset.agent_state_dim, dataset.num_pickers
    )
    
    # Unnormalize initial gripper positions
    if stats and "obs" in stats:
        obs_unnorm = unnormalize_data(initial_obs[None, :], stats["obs"])[0]
        initial_gripper_positions = extract_gripper_positions_3d(
            obs_unnorm, dataset.agent_state_dim, dataset.num_pickers
        )
    
    # Predicted actions are already unnormalized
    pred_actions_unnorm = pred_actions_np
    
    # Load image from dataset
    image = load_image_from_dataset(dataset_path, sample_idx, dataset.episode_ends, dataset.indices)
    
    # Set up camera parameters
    if cam_pos is None:
        cam_pos = DEFAULT_CAM_POS
    if cam_angle is None:
        cam_angle = DEFAULT_CAM_ANGLE
    
    cam_angle_rad = np.radians(cam_angle)
    
    # Get image dimensions
    if image is not None:
        img_h, img_w = image.shape[:2]
        frame = image.copy()
    else:
        img_h, img_w = 128, 128
        frame = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
    
    # Colors for each gripper
    gripper_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]  # Yellow, Magenta, Cyan, Green
    
    # Draw initial gripper positions
    for gripper_id in range(dataset.num_pickers):
        gripper_pos_3d = initial_gripper_positions[gripper_id]
        pixel = world_to_pixel_projection(gripper_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
        if pixel is not None:
            color = gripper_colors[gripper_id % len(gripper_colors)]
            cv2.circle(frame_bgr, pixel, 10, color, -1)
            cv2.circle(frame_bgr, pixel, 10, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Start{gripper_id}",
                       (pixel[0] + 12, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw predicted reference positions
    for gripper_id in range(dataset.num_pickers):
        ref_pos_3d = pred_ref_pos_unnorm[gripper_id]
        pixel = world_to_pixel_projection(ref_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
        if pixel is not None:
            color = gripper_colors[gripper_id % len(gripper_colors)]
            star_size = 12
            star_points = []
            for i in range(10):
                angle = i * np.pi / 5 - np.pi / 2
                r = star_size if i % 2 == 0 else star_size // 2
                star_points.append((
                    pixel[0] + int(r * np.cos(angle)),
                    pixel[1] + int(r * np.sin(angle))
                ))
            pts = np.array(star_points, np.int32)
            cv2.fillPoly(frame_bgr, [pts], color)
            cv2.polylines(frame_bgr, [pts], True, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Ref{gripper_id}",
                       (pixel[0] + 15, pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw predicted action sequence with color gradient
    action_horizon = len(pred_actions_unnorm)
    for gripper_id in range(dataset.num_pickers):
        action_start_idx = gripper_id * 4
        base_color = gripper_colors[gripper_id % len(gripper_colors)]
        
        prev_pos_3d = initial_gripper_positions[gripper_id].copy()
        for t in range(action_horizon):
            # Get translation from action
            dx, dy, dz = pred_actions_unnorm[t, action_start_idx:action_start_idx + 3]
            next_pos_3d = prev_pos_3d + np.array([dx, dy, dz])
            
            # Color gradient: darker to brighter
            alpha = t / max(action_horizon - 1, 1)
            color = tuple(int(c * (0.5 + 0.5 * alpha)) for c in base_color)
            
            pixel = world_to_pixel_projection(next_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
            if pixel is not None:
                radius = int(4 + 2 * alpha)
                cv2.circle(frame_bgr, pixel, radius, color, -1)
                if t > 0:
                    prev_pixel = world_to_pixel_projection(prev_pos_3d, cam_pos, cam_angle_rad, img_h, img_w, fov)
                    if prev_pixel is not None:
                        cv2.line(frame_bgr, prev_pixel, pixel, color, 2)
            
            prev_pos_3d = next_pos_3d
    
    # Convert back to RGB for saving
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.imsave(out_path, frame_rgb)
    
    return {
        "image_path": out_path,
        "pred_ref_pos": pred_ref_pos_unnorm,
        "pred_actions": pred_actions_unnorm,
        "initial_gripper_positions": initial_gripper_positions,
    }
