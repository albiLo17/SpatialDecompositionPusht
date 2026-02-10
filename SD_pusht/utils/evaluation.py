"""Evaluation utilities for diffusion and flow matching policy."""

import os
import numpy as np
import torch
from tqdm import tqdm
from gymnasium.vector import SyncVectorEnv
import imageio.v3 as iio
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from SD_pusht.utils.environment import make_env, apply_legacy_state
    GYM_PUSHT_AVAILABLE = True
except ImportError:
    GYM_PUSHT_AVAILABLE = False
    make_env = None
    apply_legacy_state = None

from SD_pusht.utils.normalization import normalize_data, unnormalize_data
from SD_pusht.utils.visualization import tile_images


def world_to_pixel(world_pos, swap_xy=False, flip_y=False):
    """Convert world coordinates [0, 512] to pixel coordinates [0, 512]
    
    This function matches the coordinate mapping fix from visualize_dataset_reference_frames.py.
    PushT uses [0, 512] for both x and y in world coordinates, and the rendered image is 512x512.
    
    Args:
        world_pos: [x, y] in world coordinates
        swap_xy: If True, swap x and y (default: False)
        flip_y: If True, flip y-axis for image coordinates where y=0 is at top (default: True)
    
    Returns:
        (x_pixel, y_pixel) tuple
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


def _call_render_all(venv):
    """Call render on a VectorEnv and return list of frames (length = num_envs)."""
    # prefer venv.call("render") if available (calls method on each sub-env)
    if hasattr(venv, "call"):
        try:
            return venv.call("render")
        except Exception:
            pass
    # fallback to venv.render() which may return (B,H,W,C) array
    try:
        out = venv.render()
        # if array, convert to list of frames
        if isinstance(out, np.ndarray):
            return [out[i] for i in range(out.shape[0])]
        return out
    except Exception:
        return [None] * getattr(venv, "num_envs", 0)


def evaluate_model(
    model,
    stats,
    out_path="vis_all_envs.mp4",
    num_envs=16,
    max_steps=300,
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8,
    device=None,
    use_flow_matching=False,
    save_video=True,
):
    """
    Evaluate a diffusion or flow matching model on `num_envs` parallel PushTEnv environments.
    Saves only 1 video (from the first environment) to save memory.

    Args:
      model: a Diffusion or FlowMatching model already loaded and in eval() mode.
      stats: dataset stats dict required by normalize_data/unnormalize_data.
      out_path: path to write mp4 video (only saves 1 video from first env).
      num_envs: number of parallel envs.
      max_steps: max environment steps (per-env) to run.
      pred_horizon/obs_horizon/action_horizon: horizons used by the model/dataset.
      device: torch.device or None (will infer from model parameters).
      use_flow_matching: whether the model is a FlowMatching model (True) or Diffusion model (False).
      save_video: whether to save video to disk.
    Returns:
      dict with summary: mean_score, max_score, success_rate, steps_done, video_frames
      video_frames: numpy array of shape (T, H, W, C) for wandb logging (from first env only)
    """
    if not GYM_PUSHT_AVAILABLE:
        raise ImportError("gym_pusht is not installed. This function requires gym_pusht for PushT environment evaluation. "
                         "Install it with: pip install gym-pusht")
    
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create vectorized envs
    env_fns = [make_env(render_mode="rgb_array", legacy=True, seed=100000 + i) for i in range(num_envs)]
    venv = SyncVectorEnv(env_fns)
    obs, infos = venv.reset()

    # rolling obs buffer: (B, obs_horizon, obs_dim)
    obs_buf = np.repeat(obs[:, None, :], repeats=obs_horizon, axis=1)

    # bookkeeping
    rewards_sum = np.zeros((num_envs,), dtype=np.float32)
    successes = np.zeros((num_envs,), dtype=bool)
    steps_done = 0
    max_single_reward_per_env = np.full((num_envs,), -np.inf, dtype=np.float32)  # Track max single-step reward per environment

    # visualization frames: list[T] where each element is list[B] frames
    # We only save frames from the first environment to save memory
    frames_all = []
    frames_single_env = []  # Only frames from first env for wandb
    initial_frames = _call_render_all(venv)
    if any(f is not None for f in initial_frames):
        frames_all.append([np.asarray(f, dtype=np.uint8) if f is not None else None for f in initial_frames])
        # Save first env frame for single video
        if initial_frames[0] is not None:
            frames_single_env.append(np.asarray(initial_frames[0], dtype=np.uint8))

    alive = np.ones((num_envs,), dtype=bool)

    pbar = tqdm(total=max_steps, desc="Eval PushTStateEnv (vectorized)")
    try:
        while steps_done < max_steps:
            B = num_envs

            # prepare conditional observation
            nobs = normalize_data(obs_buf, stats=stats["obs"])
            nobs_t = torch.as_tensor(nobs, device=device, dtype=torch.float32)

            with torch.no_grad():
                # Use model's get_action method which returns a dict with unnormalized actions
                action_dict = model.get_action(
                    obs_seq=nobs_t,
                    action_stats=stats.get("action") if stats else None,
                    reference_pos_stats={
                        'min': stats["obs"]['min'][0:2],
                        'max': stats["obs"]['max'][0:2],
                    } if stats and "obs" in stats else None,
                )
                # Actions are already unnormalized
                action_chunk = action_dict["actions"].detach().cpu().numpy()  # (B, H, A) - unnormalized

            t = 0
            while t < action_horizon and steps_done < max_steps:
                a_t = action_chunk[:, t, :]  # (B, A)

                # mask actions for finished envs
                a_t = a_t * alive[:, None]

                obs, rew, terminated, truncated, infos = venv.step(a_t)

                # gather success flags robustly
                if isinstance(infos, dict) and "is_success" in infos:
                    succ_flags = np.array(infos["is_success"], dtype=bool)
                else:
                    # assume list of dicts
                    try:
                        succ_flags = np.array([inf.get("is_success", False) for inf in infos], dtype=bool)
                    except Exception:
                        succ_flags = np.zeros((num_envs,), dtype=bool)

                dones = np.logical_or(terminated, truncated)

                # update successes (any time an env reports success)
                successes |= succ_flags

                # accumulate rewards only for alive envs
                rewards_sum += (rew.astype(np.float32) * alive)
                
                # track maximum single-step reward per environment (only for alive envs)
                if np.any(alive):
                    alive_indices = np.where(alive)[0]
                    for idx in alive_indices:
                        max_single_reward_per_env[idx] = max(max_single_reward_per_env[idx], float(rew[idx]))

                # update rolling obs buffer only for still-alive envs
                if np.any(alive):
                    obs_buf_alive_new = np.concatenate([obs_buf[alive, 1:, :], obs[alive][:, None, :]], axis=1)
                    obs_buf[alive] = obs_buf_alive_new

                # mark newly finished envs as dead
                alive = np.logical_and(alive, np.logical_not(dones))

                # render frames for all envs and record
                frames = _call_render_all(venv)
                if any(f is not None for f in frames):
                    frames_all.append([np.asarray(f, dtype=np.uint8) if f is not None else None for f in frames])
                    # Save first env frame for single video (for wandb)
                    if frames[0] is not None:
                        frames_single_env.append(np.asarray(frames[0], dtype=np.uint8))

                steps_done += 1
                t += 1
                pbar.update(1)
                pbar.set_postfix(mean_reward=float(np.mean(rew)))

                if steps_done >= max_steps or not np.any(alive):
                    break
    finally:
        pbar.close()

    # summary
    mean_score = float(np.mean(rewards_sum))
    max_score = float(np.max(rewards_sum))  # Max cumulative reward
    # Average of max single-step reward per environment (only count environments that received rewards)
    valid_max_rewards = max_single_reward_per_env[max_single_reward_per_env != -np.inf]
    mean_max_single_reward = float(np.mean(valid_max_rewards)) if len(valid_max_rewards) > 0 else 0.0
    success_count = int(successes.sum())
    success_rate = success_count / float(num_envs)
    print("Mean cumulative score across envs:", mean_score)
    print("Max cumulative score across envs:", max_score)
    print("Mean max single-step reward (avg of max per env):", mean_max_single_reward)
    print(f"Success rate: {success_count}/{num_envs} ({success_rate:.2%})")

    # Prepare video from first environment only (for memory efficiency)
    video_frames = None
    if len(frames_single_env) > 0:
        # Stack frames into numpy array: (T, H, W, C)
        video_frames = np.stack(frames_single_env, axis=0)
        
        # Save single video to disk if requested
        if save_video:
            # Create directory if it doesn't exist
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            iio.imwrite(out_path, video_frames, fps=30)
            print(f"Saved single environment video to {out_path}")

    venv.close()

    return {
        "mean_score": mean_score,
        "max_score": max_score,  # Max cumulative reward
        "mean_max_single_reward": mean_max_single_reward,  # Average of max single-step reward per environment
        "success_count": success_count,
        "success_rate": success_rate,
        "steps_done": steps_done,
        "video_frames": video_frames,  # (T, H, W, C) numpy array (optional, for debugging)
        "video_path": out_path if save_video and video_frames is not None else None,  # Path to saved video for wandb
    }


def visualize_training_trajectory(
    model,
    dataset,
    stats,
    sample_idx=0,
    out_path="training_trajectory_vis.png",
    device=None,
):
    """
    Visualize a training trajectory with ground truth and predicted reference positions and actions.
    
    Args:
        model: LocalFlowPolicy2D model
        dataset: PushTSegmentedDataset instance
        stats: Dataset statistics for unnormalization
        sample_idx: Index of the sample to visualize
        out_path: Path to save the visualization
        device: Device to run model on
    
    Returns:
        Dictionary with visualization information
    """
    if not GYM_PUSHT_AVAILABLE:
        raise ImportError("gym_pusht is not installed. This function requires gym_pusht for PushT environment visualization. "
                         "Install it with: pip install gym-pusht")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    obs_seq = torch.from_numpy(sample['obs']).float().unsqueeze(0).to(device)  # (1, obs_horizon, obs_dim)
    gt_actions = torch.from_numpy(sample['action']).float().unsqueeze(0).to(device)  # (1, pred_horizon, act_dim)
    gt_ref_pos = torch.from_numpy(sample['reference_pos']).float().unsqueeze(0).to(device)  # (1, 2)
    
    # Get initial state (first observation)
    initial_state = sample['obs'][0]  # (obs_dim,)
    
    # Unnormalize initial state for rendering
    initial_state_unnorm = unnormalize_data(initial_state[None, :], stats["obs"])[0]
    # State: [agent_x, agent_y, block_x, block_y, block_angle]
    
    # Get predictions from model
    action_stats = stats.get("action", None) if stats else None
    reference_pos_stats = None
    if stats and "obs" in stats:
        reference_pos_stats = {
            'min': stats["obs"]['min'][0:2],  # First 2 dims are agent position
            'max': stats["obs"]['max'][0:2],
        }
    
    with torch.no_grad():
        predictions = model.get_action(
            obs_seq,
            reference_position=gt_ref_pos,
            action_stats=action_stats,
            reference_pos_stats=reference_pos_stats,
        )
        pred_ref_pos = predictions["reference_pos"]  # (1, 2) - normalized
        pred_actions = predictions["actions"]  # (1, action_horizon, act_dim) - UNNORMALIZED world actions
        pred_local_actions = predictions.get("local_actions", pred_actions)  # (1, action_horizon, act_dim) - normalized local actions
    
    # Convert to numpy
    pred_ref_pos_np = pred_ref_pos.cpu().numpy()[0]  # (2,) - normalized
    pred_actions_np = pred_actions.cpu().numpy()[0]  # (action_horizon, act_dim) - UNNORMALIZED world actions
    gt_ref_pos_np = gt_ref_pos.cpu().numpy()[0]  # (2,) - normalized
    gt_actions_np = gt_actions.cpu().numpy()[0]  # (pred_horizon, act_dim) - normalized world actions
    
    # Unnormalize predictions
    # Use agent position stats for reference positions (matching visualize_dataset_reference_frames.py)
    # Reference position is the agent position at a future timestep, so it uses the same normalization as agent positions
    agent_stats = {
        'min': stats["obs"]['min'][0:2],
        'max': stats["obs"]['max'][0:2],
    } if stats and "obs" in stats else None
    
    # Always unnormalize ground truth reference position using agent stats
    # (matching visualize_dataset_reference_frames.py which assumes reference_pos from dataset is normalized)
    gt_ref_pos_unnorm = unnormalize_data(gt_ref_pos_np[None, :], agent_stats)[0] if agent_stats else gt_ref_pos_np
    
    # Model predictions are always normalized (output from model is in [-1, 1] range)
    # Use agent stats for denormalization (same as ground truth)
    pred_ref_pos_unnorm = unnormalize_data(pred_ref_pos_np[None, :], agent_stats)[0] if agent_stats else pred_ref_pos_np
    
    if action_stats is not None:
        # Predicted actions are already unnormalized (from get_action)
        pred_actions_unnorm = pred_actions_np
        # Unnormalize GT actions for comparison
        gt_actions_unnorm = unnormalize_data(gt_actions_np[:len(pred_actions_np)], action_stats)
    else:
        pred_actions_unnorm = pred_actions_np
        gt_actions_unnorm = gt_actions_np[:len(pred_actions_np)]
    
    # Create environment and render initial state
    env_fn = make_env(render_mode="rgb_array", legacy=True)
    env = env_fn()
    env.reset(options={"reset_to_state": initial_state_unnorm})
    frame = env.render()  # (H, W, 3) uint8
    
    # Resize frame if needed (PushT renders at 512x512, but we might want larger)
    if frame.shape[0] != 512 or frame.shape[1] != 512:
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw ground truth reference position (green)
    gt_ref_pos_pixel = world_to_pixel(gt_ref_pos_unnorm, swap_xy=False, flip_y=False)
    cv2.circle(frame_bgr, gt_ref_pos_pixel, 12, (0, 255, 0), -1)  # Green filled circle
    cv2.putText(frame_bgr, "GT Ref", 
                (gt_ref_pos_pixel[0] + 15, gt_ref_pos_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw predicted reference position (red)
    pred_ref_pos_pixel = world_to_pixel(pred_ref_pos_unnorm, swap_xy=False, flip_y=False)
    cv2.circle(frame_bgr, pred_ref_pos_pixel, 12, (0, 0, 255), -1)  # Red filled circle
    cv2.putText(frame_bgr, "Pred Ref", 
                (pred_ref_pos_pixel[0] + 15, pred_ref_pos_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw predicted action sequence with colors (blue to yellow gradient)
    action_horizon = len(pred_actions_unnorm)
    try:
        cmap = plt.colormaps['viridis']  # Color map from blue to yellow (new matplotlib API)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('viridis')  # Fallback for older matplotlib
    
    for t in range(action_horizon):
        # Get color based on timestep (0 = blue, action_horizon-1 = yellow)
        color_norm = t / max(action_horizon - 1, 1)
        color_rgb = cmap(color_norm)[:3]  # Get RGB (ignore alpha)
        color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])  # Convert to BGR and scale to 0-255
        
        # Action is target agent position
        action_pos = pred_actions_unnorm[t]
        action_pixel = world_to_pixel(action_pos, swap_xy=False, flip_y=False)
        
        # Draw circle for action
        radius = 6
        cv2.circle(frame_bgr, action_pixel, radius, color_bgr, -1)
        
        # Draw line from previous action (if not first)
        if t > 0:
            prev_action_pos = pred_actions_unnorm[t-1]
            prev_action_pixel = world_to_pixel(prev_action_pos, swap_xy=False, flip_y=False)
            cv2.line(frame_bgr, prev_action_pixel, action_pixel, color_bgr, 2)
        
        # Draw timestep number
        cv2.putText(frame_bgr, str(t), 
                    (action_pixel[0] + 8, action_pixel[1] + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw line from initial agent position to first action
    initial_agent_pos = initial_state_unnorm[:2]
    initial_agent_pixel = world_to_pixel(initial_agent_pos, swap_xy=False, flip_y=False)
    first_action_pixel = world_to_pixel(pred_actions_unnorm[0], swap_xy=False, flip_y=False)
    cv2.line(frame_bgr, initial_agent_pixel, first_action_pixel, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Convert back to RGB for saving
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.imsave(out_path, frame_rgb)
    
    env.close()
    
    return {
        "image_path": out_path,
        "gt_ref_pos": gt_ref_pos_unnorm,
        "pred_ref_pos": pred_ref_pos_unnorm,
        "pred_actions": pred_actions_unnorm,
        "initial_state": initial_state_unnorm,
    }


def _draw_particles_on_frame(frame, all_particles, selected_index, stats=None, 
                             particle_color=(100, 100, 255), selected_color=(0, 255, 0), 
                             particle_radius=5, selected_radius=10):
    """Draw all particles and highlight the selected one on frame.
    
    Args:
        frame: Image frame, shape (H, W, 3) as uint8 (RGB)
        all_particles: All particle positions in normalized coordinates, shape (num_particles, 2)
        selected_index: Index of selected particle (None if no selection)
        stats: Dataset stats for unnormalization (required for proper mapping)
        particle_color: RGB color for regular particles (will convert to BGR for OpenCV)
        selected_color: RGB color for selected particle (will convert to BGR for OpenCV)
        particle_radius: Radius of regular particle markers
        selected_radius: Radius of selected particle marker
    
    Returns:
        Frame with particles drawn (RGB)
    """
    if frame is None or all_particles is None:
        return frame
    
    frame = frame.copy()
    H, W = frame.shape[:2]
    
    # Unnormalize particles using stats
    if stats is not None and "obs" in stats:
        agent_stats = {
            'min': stats["obs"]['min'][0:2],
            'max': stats["obs"]['max'][0:2],
        }
        # Unnormalize from [-1, 1] back to original range [0, 512]
        particles_unnorm = unnormalize_data(all_particles, agent_stats)
    else:
        particles_unnorm = all_particles
    
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Convert particle colors from RGB to BGR
    particle_color_bgr = (particle_color[2], particle_color[1], particle_color[0])
    selected_color_bgr = (selected_color[2], selected_color[1], selected_color[0])
    
    # Draw all particles
    num_particles = len(particles_unnorm)
    for i, particle_pos in enumerate(particles_unnorm):
        x_pixel, y_pixel = world_to_pixel(particle_pos, swap_xy=False, flip_y=False)
        
        # Scale to actual image dimensions if different from 512
        x_pixel = int(x_pixel * (W / 512.0))
        y_pixel = int(y_pixel * (H / 512.0))
        
        # Clamp to image bounds
        x_pixel = max(0, min(W - 1, x_pixel))
        y_pixel = max(0, min(H - 1, y_pixel))
        
        # Draw particle
        if selected_index is not None and i == selected_index:
            # Highlight selected particle (green, larger)
            cv2.circle(frame_bgr, (x_pixel, y_pixel), selected_radius, selected_color_bgr, -1)
            cv2.circle(frame_bgr, (x_pixel, y_pixel), selected_radius, (255, 255, 255), 2)  # White border
            # Add label
            cv2.putText(frame_bgr, f"S{i}", 
                       (x_pixel + selected_radius + 2, y_pixel),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, selected_color_bgr, 2)
        else:
            # Regular particle (blue, smaller)
            cv2.circle(frame_bgr, (x_pixel, y_pixel), particle_radius, particle_color_bgr, -1)
            cv2.circle(frame_bgr, (x_pixel, y_pixel), particle_radius, (255, 255, 255), 1)  # White border
    
    # Convert back to RGB
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame


def _draw_reference_position(frame, ref_pos, stats=None, color=(255, 0, 0), radius=8):
    """Draw reference position on frame.
    
    Args:
        frame: Image frame, shape (H, W, 3) as uint8
        ref_pos: Reference position in normalized coordinates, shape (2,)
        stats: Dataset stats for unnormalization (required for proper mapping)
        color: RGB color for the marker (BGR for OpenCV)
        radius: Radius of the marker circle
    
    Returns:
        Frame with reference position drawn
    """
    if frame is None:
        return None
    
    frame = frame.copy()
    H, W = frame.shape[:2]
    
    # Unnormalize reference position using stats
    # PushT coordinates are in [0, 512] range, but normalized to [-1, 1]
    # Use agent position stats for reference positions (matching visualize_dataset_reference_frames.py)
    # Reference position is the agent position at a future timestep, so it uses the same normalization
    if stats is not None and "obs" in stats:
        # Extract stats for agent position (indices 0:2 in observation)
        agent_stats = {
            'min': stats["obs"]['min'][0:2],  # agent_x, agent_y min
            'max': stats["obs"]['max'][0:2],  # agent_x, agent_y max
        }
        # Unnormalize from [-1, 1] back to original range [0, 512]
        ref_pos_unnorm = unnormalize_data(ref_pos[None, :], agent_stats)[0]
    else:
        # Fallback: assume coordinates are already in [0, 512] range
        ref_pos_unnorm = ref_pos
    
    # PushT uses coordinates in [0, 512] range
    # Map to pixel coordinates: [0, 512] -> [0, W] and [0, H]
    # Use consistent coordinate mapping with flip_y=True (matching visualize_dataset_reference_frames.py)
    x_pixel, y_pixel = world_to_pixel(ref_pos_unnorm, swap_xy=False, flip_y=False)
    # Scale to actual image dimensions if different from 512
    if W != 512:
        x_pixel = int((x_pixel / 512.0) * W)
    if H != 512:
        y_pixel = int((y_pixel / 512.0) * H)
    
    # Clamp to image bounds
    x_pixel = np.clip(x_pixel, 0, W - 1)
    y_pixel = np.clip(y_pixel, 0, H - 1)
    
    # Draw filled circle
    cv2.circle(frame, (x_pixel, y_pixel), radius, color, -1)
    # Draw crosshair for better visibility
    crosshair_size = radius * 2
    cv2.line(frame, 
             (x_pixel - crosshair_size, y_pixel), 
             (x_pixel + crosshair_size, y_pixel), 
             color, 2)
    cv2.line(frame, 
             (x_pixel, y_pixel - crosshair_size), 
             (x_pixel, y_pixel + crosshair_size), 
             color, 2)
    
    return frame


def evaluate_local_flow_2d(
    model,
    stats,
    out_path="vis_local_flow_2d.mp4",
    num_envs=16,
    max_steps=300,
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8,
    device=None,
    save_video=True,
    eval_seed=None,
    eval_input_noise_std=0.0,
    k_clusters=None,
    visualize_particles=False,
    particles_output_dir=None,
):
    """
    Evaluate a LocalFlowPolicy2D model on `num_envs` parallel PushTEnv environments.
    Visualizes the predicted reference position in the video.
    
    Args:
      model: a LocalFlowPolicy2D model already loaded and in eval() mode.
      stats: dataset stats dict required by normalize_data/unnormalize_data.
      out_path: path to write mp4 video (only saves 1 video from first env).
      num_envs: number of parallel envs.
      max_steps: max environment steps (per-env) to run.
      pred_horizon/obs_horizon/action_horizon: horizons used by the model/dataset.
      device: torch.device or None (will infer from model parameters).
      save_video: whether to save video to disk.
      eval_seed: Random seed for environment initialization (None = use fixed seeds).
      eval_input_noise_std: Standard deviation of Gaussian noise to add to observations during evaluation.
        This is separate from the model's input_noise_std parameter (which should be 0 for evaluation).
        Default: 0.0 (no noise).
      k_clusters: Optional number of nearest neighbors for KNN aggregation (only used if particles_aggregation="knn").
        If None, uses default k = max(1, P // 2) where P is num_particles. Default: None (backward compatible).
      visualize_particles: If True, visualize all particles and selected one for the first environment.
        Saves images at each action prediction to particles_output_dir. Default: False (backward compatible).
      particles_output_dir: Directory to save particle visualization images (only used if visualize_particles=True).
    Returns:
      dict with summary: mean_score, max_score, success_rate, steps_done, video_frames
      video_frames: numpy array of shape (T, H, W, C) for wandb logging (from first env only)
    """
    if not GYM_PUSHT_AVAILABLE:
        raise ImportError("gym_pusht is not installed. This function requires gym_pusht for PushT environment evaluation. "
                         "Install it with: pip install gym-pusht")
    
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create vectorized envs with randomized seeds
    if eval_seed is not None:
        # Use eval_seed to randomize environment resets
        rng = np.random.RandomState(eval_seed)
        env_seeds = [rng.randint(0, 2**31) for _ in range(num_envs)]
    else:
        # Default: use fixed seeds based on env index
        env_seeds = [100000 + i for i in range(num_envs)]
    
    env_fns = [make_env(render_mode="rgb_array", legacy=True, seed=seed) for seed in env_seeds]
    venv = SyncVectorEnv(env_fns)
    obs, infos = venv.reset()

    # rolling obs buffer: (B, obs_horizon, obs_dim)
    obs_buf = np.repeat(obs[:, None, :], repeats=obs_horizon, axis=1)

    # bookkeeping
    rewards_sum = np.zeros((num_envs,), dtype=np.float32)
    successes = np.zeros((num_envs,), dtype=bool)
    steps_done = 0
    max_single_reward_per_env = np.full((num_envs,), -np.inf, dtype=np.float32)
    
    # Store reference positions for visualization (only for first env)
    current_ref_pos = None
    action_step_counter = 0  # Track steps within current action horizon
    
    # Particle visualization (only for first env)
    particle_step_counter = 0  # Track steps for particle visualization
    stored_particles_info = None  # Store particles info for current step
    max_variance_info = {'value': -1.0, 'image': None}  # Track maximum variance seen (use dict for mutability)
    if visualize_particles and particles_output_dir:
        os.makedirs(particles_output_dir, exist_ok=True)

    # visualization frames: list[T] where each element is list[B] frames
    # We only save frames from the first environment to save memory
    frames_all = []
    frames_single_env = []  # Only frames from first env for wandb
    initial_frames = _call_render_all(venv)
    if any(f is not None for f in initial_frames):
        frames_all.append([np.asarray(f, dtype=np.uint8) if f is not None else None for f in initial_frames])
        # Save first env frame for single video (without ref pos initially)
        if initial_frames[0] is not None:
            frames_single_env.append(np.asarray(initial_frames[0], dtype=np.uint8))

    alive = np.ones((num_envs,), dtype=bool)

    pbar = tqdm(total=max_steps, desc="Eval LocalFlowPolicy2D (vectorized)")
    try:
        while steps_done < max_steps:
            B = num_envs

            # prepare conditional observation
            nobs = normalize_data(obs_buf, stats=stats["obs"])
            nobs_t = torch.as_tensor(nobs, device=device, dtype=torch.float32)

            # Get new action prediction at the start of each action horizon
            if action_step_counter == 0:
                with torch.no_grad():
                    # Use model's get_action method which returns a dict with unnormalized actions
                    reference_pos_stats = {
                        'min': stats["obs"]['min'][0:2],
                        'max': stats["obs"]['max'][0:2],
                    } if stats and "obs" in stats else None
                    action_dict = model.get_action(
                        obs_seq=nobs_t,
                        action_stats=stats.get("action") if stats else None,
                        reference_pos_stats=reference_pos_stats,
                        k_clusters=k_clusters,
                        return_particles=visualize_particles,
                    )
                    action_pred = action_dict["actions"]  # (B, action_horizon, action_dim) - UNNORMALIZED
                    ref_pos_pred = action_dict["reference_pos"]  # (B, 2) - normalized
                    
                    # Store reference position for first env (for visualization)
                    if ref_pos_pred is not None:
                        current_ref_pos = ref_pos_pred[0].detach().cpu().numpy()
                    
                    # Get particles info for visualization (first env only)
                    stored_particles_info = None
                    if visualize_particles and "particles_info" in action_dict:
                        particles_info = action_dict["particles_info"]
                        # Extract first environment's particles
                        if particles_info.get("particles") is not None:
                            # particles shape: (B, num_particles, 2)
                            particles_first_env = particles_info["particles"][0].detach().cpu().numpy()  # (num_particles, 2)
                            selected_idx = None
                            if particles_info.get("selected_index") is not None:
                                selected_idx = particles_info["selected_index"][0].item()
                            
                            # Calculate variance of particle positions
                            # Variance across both x and y dimensions
                            particle_variance = np.var(particles_first_env, axis=0).sum()  # Sum of variances in x and y
                            
                            # Store particles info for visualization after rendering
                            stored_particles_info = {
                                'particles': particles_first_env,
                                'selected_index': selected_idx,
                                'variance': particle_variance
                            }
                            particle_step_counter += 1
                    
                    # Actions are already unnormalized, just convert to numpy
                    action_chunk = action_pred.detach().cpu().numpy()  # (B, H, A) - unnormalized

            # Use the current action from the chunk
            a_t = action_chunk[:, action_step_counter, :]  # (B, A)

            # mask actions for finished envs
            a_t = a_t * alive[:, None]

            obs, rew, terminated, truncated, infos = venv.step(a_t)

            # gather success flags robustly
            if isinstance(infos, dict) and "is_success" in infos:
                succ_flags = np.array(infos["is_success"], dtype=bool)
            else:
                # assume list of dicts
                try:
                    succ_flags = np.array([inf.get("is_success", False) for inf in infos], dtype=bool)
                except Exception:
                    succ_flags = np.zeros((num_envs,), dtype=bool)

            dones = np.logical_or(terminated, truncated)

            # update successes (any time an env reports success)
            successes |= succ_flags

            # accumulate rewards only for alive envs
            rewards_sum += (rew.astype(np.float32) * alive)
            
            # track maximum single-step reward per environment (only for alive envs)
            if np.any(alive):
                alive_indices = np.where(alive)[0]
                for idx in alive_indices:
                    max_single_reward_per_env[idx] = max(max_single_reward_per_env[idx], float(rew[idx]))

            # update rolling obs buffer only for still-alive envs
            if np.any(alive):
                obs_buf_alive_new = np.concatenate([obs_buf[alive, 1:, :], obs[alive][:, None, :]], axis=1)
                obs_buf[alive] = obs_buf_alive_new

            # mark newly finished envs as dead
            alive = np.logical_and(alive, np.logical_not(dones))

            # render frames for all envs and record
            frames = _call_render_all(venv)
            if any(f is not None for f in frames):
                frames_all.append([np.asarray(f, dtype=np.uint8) if f is not None else None for f in frames])
                # Save first env frame for single video (for wandb) with reference position
                if frames[0] is not None:
                    frame_first = np.asarray(frames[0], dtype=np.uint8)
                    
                    # Draw particles if available (only at start of action horizon)
                    if visualize_particles and stored_particles_info is not None and action_step_counter == 0:
                        frame_first = _draw_particles_on_frame(
                            frame=frame_first,
                            all_particles=stored_particles_info['particles'],
                            selected_index=stored_particles_info['selected_index'],
                            stats=stats,
                        )
                        
                        # Save particle visualization image
                        if particles_output_dir:
                            particle_image_path = os.path.join(
                                particles_output_dir, 
                                f"particles_step_{particle_step_counter-1:04d}.png"
                            )
                            plt.imsave(particle_image_path, frame_first)
                            
                            # Track highest variance image
                            current_variance = stored_particles_info.get('variance', 0.0)
                            if current_variance > max_variance_info['value']:
                                max_variance_info['value'] = current_variance
                                max_variance_info['image'] = frame_first.copy()
                                # Save/update highest variance image
                                highest_variance_path = os.path.join(
                                    particles_output_dir,
                                    "highest_variance.png"
                                )
                                plt.imsave(highest_variance_path, max_variance_info['image'])
                                print(f"Updated highest variance image: variance={current_variance:.4f}")
                            
                            # Clear stored particles info after visualization
                            stored_particles_info = None
                    
                    # Draw reference position
                    frame_with_ref = _draw_reference_position(
                        frame_first,
                        current_ref_pos if current_ref_pos is not None else np.array([0.0, 0.0]),
                        stats=stats
                    )
                    frames_single_env.append(frame_with_ref if frame_with_ref is not None else frame_first)

            steps_done += 1
            action_step_counter = (action_step_counter + 1) % action_horizon
            pbar.update(1)
            pbar.set_postfix(mean_reward=float(np.mean(rew)))

            if steps_done >= max_steps or not np.any(alive):
                break
    finally:
        pbar.close()

    # summary
    mean_score = float(np.mean(rewards_sum))
    max_score = float(np.max(rewards_sum))
    valid_max_rewards = max_single_reward_per_env[max_single_reward_per_env != -np.inf]
    mean_max_single_reward = float(np.mean(valid_max_rewards)) if len(valid_max_rewards) > 0 else 0.0
    success_count = int(successes.sum())
    success_rate = success_count / float(num_envs)
    print("Mean cumulative score across envs:", mean_score)
    print("Max cumulative score across envs:", max_score)
    print("Mean max single-step reward (avg of max per env):", mean_max_single_reward)
    print(f"Success rate: {success_count}/{num_envs} ({success_rate:.2%})")

    # Prepare video from first environment only (for memory efficiency)
    video_frames = None
    if len(frames_single_env) > 0:
        # Stack frames into numpy array: (T, H, W, C)
        video_frames = np.stack(frames_single_env, axis=0)
        
        # Save single video to disk if requested
        if save_video:
            # Create directory if it doesn't exist
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            iio.imwrite(out_path, video_frames, fps=30)
            print(f"Saved single environment video with reference position visualization to {out_path}")

    venv.close()

    return {
        "mean_score": mean_score,
        "max_score": max_score,
        "mean_max_single_reward": mean_max_single_reward,
        "success_count": success_count,
        "success_rate": success_rate,
        "steps_done": steps_done,
        "video_frames": video_frames,
        "video_path": out_path if save_video and video_frames is not None else None,
    }


def visualize_position_predictions(
    model,
    dataset,
    stats,
    sample_idx=0,
    out_path="position_prediction_vis.png",
    device=None,
):
    """
    Visualize position decoder predictions vs ground truth on an image.
    
    Args:
        model: Position2DFlowDecoder model or LocalFlowPolicy2D model (if LocalFlowPolicy2D,
               will extract position_decoder and use obs_encoder for encoding)
        dataset: PushTSegmentedDataset instance
        stats: Dataset statistics for unnormalization
        sample_idx: Index of the sample to visualize
        out_path: Path to save the visualization
        device: Device to run model on
    
    Returns:
        Dictionary with visualization information
    """
    if not GYM_PUSHT_AVAILABLE:
        raise ImportError("gym_pusht is not installed. This function requires gym_pusht for PushT environment visualization. "
                         "Install it with: pip install gym-pusht")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model is LocalFlowPolicy2D (has position_decoder and obs_encoder)
    # or just Position2DFlowDecoder
    if hasattr(model, 'position_decoder') and hasattr(model, 'obs_encoder'):
        # Full model passed - extract position decoder and encoder
        position_decoder = model.position_decoder
        obs_encoder = model.obs_encoder
        model.eval()
        position_decoder.eval()
    else:
        # Direct Position2DFlowDecoder passed (backward compatibility)
        position_decoder = model
        obs_encoder = None
        position_decoder.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    obs_seq = torch.from_numpy(sample['obs']).float().unsqueeze(0).to(device)  # (1, obs_horizon, obs_dim)
    gt_ref_pos = torch.from_numpy(sample['reference_pos']).float().unsqueeze(0).to(device)  # (1, 2)
    
    # Get initial state (first observation)
    initial_state = sample['obs'][0]  # (obs_dim,)
    
    # Unnormalize initial state for rendering
    initial_state_unnorm = unnormalize_data(initial_state[None, :], stats["obs"])[0]
    
    # Get prediction from model
    obs_flat = obs_seq.flatten(start_dim=1)  # (1, obs_horizon * obs_dim)
    
    # Encode observations if encoder is available
    if obs_encoder is not None:
        obs_cond = obs_encoder(obs_flat)  # (1, obs_encoder_dim)
    else:
        obs_cond = obs_flat  # (1, obs_horizon * obs_dim)
    
    with torch.no_grad():
        pred_ref_pos = position_decoder(obs_cond, x_init=None)  # (1, 2)
    
    # Convert to numpy
    pred_ref_pos_np = pred_ref_pos.cpu().numpy()[0]  # (2,)
    gt_ref_pos_np = gt_ref_pos.cpu().numpy()[0]  # (2,)
    
    # Unnormalize predictions
    agent_stats = {
        'min': stats["obs"]['min'][0:2],
        'max': stats["obs"]['max'][0:2],
    }
    
    gt_ref_pos_unnorm = unnormalize_data(gt_ref_pos_np[None, :], agent_stats)[0]
    pred_ref_pos_unnorm = unnormalize_data(pred_ref_pos_np[None, :], agent_stats)[0]
    
    # Compute prediction error
    pred_error = np.linalg.norm(pred_ref_pos_unnorm - gt_ref_pos_unnorm)
    
    # Create environment and render initial state
    env_fn = make_env(render_mode="rgb_array", legacy=True)
    env = env_fn()
    env.reset(options={"reset_to_state": initial_state_unnorm})
    frame = env.render()  # (H, W, 3) uint8
    
    # Resize frame if needed
    if frame.shape[0] != 512 or frame.shape[1] != 512:
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw ground truth reference position (green)
    gt_ref_pos_pixel = world_to_pixel(gt_ref_pos_unnorm, swap_xy=False, flip_y=False)
    cv2.circle(frame_bgr, gt_ref_pos_pixel, 15, (0, 255, 0), -1)  # Green filled circle
    cv2.circle(frame_bgr, gt_ref_pos_pixel, 15, (0, 0, 0), 2)  # Black border
    cv2.putText(frame_bgr, "GT", 
                (gt_ref_pos_pixel[0] + 20, gt_ref_pos_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw predicted reference position (red)
    pred_ref_pos_pixel = world_to_pixel(pred_ref_pos_unnorm, swap_xy=False, flip_y=False)
    cv2.circle(frame_bgr, pred_ref_pos_pixel, 15, (0, 0, 255), -1)  # Red filled circle
    cv2.circle(frame_bgr, pred_ref_pos_pixel, 15, (0, 0, 0), 2)  # Black border
    cv2.putText(frame_bgr, "Pred", 
                (pred_ref_pos_pixel[0] + 20, pred_ref_pos_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw line connecting GT and prediction
    cv2.line(frame_bgr, gt_ref_pos_pixel, pred_ref_pos_pixel, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Add text with error
    error_text = f"Error: {pred_error:.2f} pixels"
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Convert back to RGB for saving
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.imsave(out_path, frame_rgb)
    
    env.close()
    
    return {
        "image_path": out_path,
        "gt_ref_pos": gt_ref_pos_unnorm,
        "pred_ref_pos": pred_ref_pos_unnorm,
        "pred_error": pred_error,
        "initial_state": initial_state_unnorm,
    }


def visualize_action_predictions(
    model,
    dataset,
    stats,
    sample_idx=0,
    out_path="action_prediction_vis.png",
    device=None,
):
    """
    Visualize action policy predictions vs ground truth actions on an image.
    
    Args:
        model: LocalFlowPolicy2D model (action policy only)
        dataset: PushTSegmentedDataset instance
        stats: Dataset statistics for unnormalization
        sample_idx: Index of the sample to visualize
        out_path: Path to save the visualization
        device: Device to run model on
    
    Returns:
        Dictionary with visualization information
    """
    if not GYM_PUSHT_AVAILABLE:
        raise ImportError("gym_pusht is not installed. This function requires gym_pusht for PushT environment visualization. "
                         "Install it with: pip install gym-pusht")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    obs_seq = torch.from_numpy(sample['obs']).float().unsqueeze(0).to(device)  # (1, obs_horizon, obs_dim)
    gt_actions = torch.from_numpy(sample['action']).float().unsqueeze(0).to(device)  # (1, pred_horizon, act_dim)
    gt_ref_pos = torch.from_numpy(sample['reference_pos']).float().unsqueeze(0).to(device)  # (1, 2)
    
    # Get initial state (first observation)
    initial_state = sample['obs'][0]  # (obs_dim,)
    
    # Unnormalize initial state for rendering
    initial_state_unnorm = unnormalize_data(initial_state[None, :], stats["obs"])[0]
    
    # Get predictions from model
    action_stats = stats.get("action", None) if stats else None
    reference_pos_stats = None
    if stats and "obs" in stats:
        reference_pos_stats = {
            'min': stats["obs"]['min'][0:2],  # First 2 dims are agent position
            'max': stats["obs"]['max'][0:2],
        }
    
    with torch.no_grad():
        predictions = model.get_action(
            obs_seq, 
            reference_position=gt_ref_pos,
            action_stats=action_stats,
            reference_pos_stats=reference_pos_stats,
        )
        pred_world_actions = predictions["actions"]  # (1, action_horizon, act_dim) - UNNORMALIZED world actions
        pred_ref_pos = predictions.get("reference_pos", gt_ref_pos)  # (1, 2) - normalized reference position
    
    # Convert to numpy
    pred_world_actions_np = pred_world_actions.cpu().numpy()[0]  # (action_horizon, act_dim) - UNNORMALIZED world actions
    gt_actions_np = gt_actions.cpu().numpy()[0]  # (pred_horizon, act_dim) - normalized world actions
    pred_ref_pos_np = pred_ref_pos.cpu().numpy()[0]  # (2,) - normalized reference position
    
    # Unnormalize predictions and GT for visualization
    agent_stats = {
        'min': stats["obs"]['min'][0:2],
        'max': stats["obs"]['max'][0:2],
    } if stats and "obs" in stats else None
    gt_ref_pos_unnorm = unnormalize_data(gt_ref_pos.cpu().numpy()[0][None, :], agent_stats)[0] if agent_stats else gt_ref_pos.cpu().numpy()[0]
    
    if action_stats is not None:
        # Predicted world actions are already unnormalized (from get_action)
        pred_actions_unnorm = pred_world_actions_np
        
        # Unnormalize GT world actions for comparison
        gt_actions_unnorm = unnormalize_data(gt_actions_np[:len(pred_actions_unnorm)], action_stats)
        
        # Unnormalize reference position for visualization
        pred_ref_pos_unnorm = unnormalize_data(pred_ref_pos_np[None, :], agent_stats)[0] if agent_stats else pred_ref_pos_np
    else:
        # Fallback if no stats
        pred_actions_unnorm = pred_world_actions_np
        gt_actions_unnorm = gt_actions_np[:len(pred_actions_unnorm)]
        pred_ref_pos_unnorm = pred_ref_pos_np
    
    # Compute action error
    action_error = np.mean(np.linalg.norm(pred_actions_unnorm - gt_actions_unnorm, axis=1))
    
    # Create environment and render initial state
    env_fn = make_env(render_mode="rgb_array", legacy=True)
    env = env_fn()
    env.reset(options={"reset_to_state": initial_state_unnorm})
    frame = env.render()  # (H, W, 3) uint8
    
    # Resize frame if needed
    if frame.shape[0] != 512 or frame.shape[1] != 512:
        frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Convert to BGR for OpenCV drawing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Draw reference position (yellow)
    ref_pos_pixel = world_to_pixel(gt_ref_pos_unnorm, swap_xy=False, flip_y=False)
    cv2.circle(frame_bgr, ref_pos_pixel, 12, (0, 255, 255), -1)  # Yellow filled circle
    cv2.putText(frame_bgr, "Ref", 
                (ref_pos_pixel[0] + 15, ref_pos_pixel[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw ground truth action sequence (green)
    action_horizon = len(gt_actions_unnorm)
    for t in range(action_horizon):
        action_pos = gt_actions_unnorm[t]
        action_pixel = world_to_pixel(action_pos, swap_xy=False, flip_y=False)
        cv2.circle(frame_bgr, action_pixel, 8, (0, 255, 0), -1)  # Green circle
        if t > 0:
            prev_action_pos = gt_actions_unnorm[t-1]
            prev_action_pixel = world_to_pixel(prev_action_pos, swap_xy=False, flip_y=False)
            cv2.line(frame_bgr, prev_action_pixel, action_pixel, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"GT{t}", 
                    (action_pixel[0] + 10, action_pixel[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw predicted action sequence (red)
    for t in range(len(pred_actions_unnorm)):
        action_pos = pred_actions_unnorm[t]
        action_pixel = world_to_pixel(action_pos, swap_xy=False, flip_y=False)
        cv2.circle(frame_bgr, action_pixel, 8, (0, 0, 255), -1)  # Red circle
        if t > 0:
            prev_action_pos = pred_actions_unnorm[t-1]
            prev_action_pixel = world_to_pixel(prev_action_pos, swap_xy=False, flip_y=False)
            cv2.line(frame_bgr, prev_action_pixel, action_pixel, (0, 0, 255), 2)
        cv2.putText(frame_bgr, f"P{t}", 
                    (action_pixel[0] + 10, action_pixel[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Draw lines from initial agent position
    initial_agent_pos = initial_state_unnorm[:2]
    initial_agent_pixel = world_to_pixel(initial_agent_pos, swap_xy=False, flip_y=False)
    if len(gt_actions_unnorm) > 0:
        first_gt_action_pixel = world_to_pixel(gt_actions_unnorm[0], swap_xy=False, flip_y=False)
        cv2.line(frame_bgr, initial_agent_pixel, first_gt_action_pixel, (0, 255, 0), 2, cv2.LINE_AA)
    if len(pred_actions_unnorm) > 0:
        first_pred_action_pixel = world_to_pixel(pred_actions_unnorm[0], swap_xy=False, flip_y=False)
        cv2.line(frame_bgr, initial_agent_pixel, first_pred_action_pixel, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Add text with error
    error_text = f"Action Error: {action_error:.2f} pixels"
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_bgr, error_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Convert back to RGB for saving
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Save visualization
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.imsave(out_path, frame_rgb)
    
    env.close()
    
    return {
        "image_path": out_path,
        "gt_actions": gt_actions_unnorm,
        "pred_actions": pred_actions_unnorm,
        "action_error": action_error,
        "initial_state": initial_state_unnorm,
    }

