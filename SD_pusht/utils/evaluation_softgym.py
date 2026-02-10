"""Evaluation utilities for SoftGym environments."""

import os
import numpy as np
import torch
from tqdm import tqdm
import imageio.v3 as iio
from copy import deepcopy
from typing import Optional, Dict, Any

from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict
from softgym.utils.normalized_env import normalize
from SD_pusht.utils.normalization import normalize_data, unnormalize_data


def build_softgym_env(env_name: str, env_kwargs: Optional[Dict[str, Any]] = None, seed: Optional[int] = None, num_envs: int = 1):
    """Build a SoftGym environment.
    
    Follows the same pattern as generate_expert_trajs.py:
    - Creates environment with num_variations set in kwargs
    - Environment generates variations at initialization
    - Use reset(config_id=i) to select specific variation
    
    Args:
        env_name: Name of the SoftGym environment (e.g., "RopeFlatten")
        env_kwargs: Optional dictionary of environment kwargs to override defaults
                   Should include num_variations if you want specific number
        seed: Optional random seed
        num_envs: Number of environments to build

    Returns:
        Normalized SoftGym environment
    """
    if env_name not in env_arg_dict:
        raise ValueError(f"Unknown env_name '{env_name}'. Available: {list(env_arg_dict.keys())}")
    
    kwargs = deepcopy(env_arg_dict[env_name])
    kwargs["use_cached_states"] = True
    kwargs["save_cached_states"] = False
    kwargs["render"] = True
    kwargs["headless"] = True  # Headless for evaluation
    kwargs["observation_mode"] = "key_point"  # Use state observations
    
    # force the number of variations to be equal to the number of environments
    kwargs['num_variations'] = num_envs
    
    # Override with provided kwargs (including num_variations)
    if env_kwargs:
        kwargs.update(env_kwargs)
        
    print(kwargs)
    
    # Create base environment (this generates num_variations variations at init)
    base_env = SOFTGYM_ENVS[env_name](**kwargs)
    env = normalize(base_env)
    
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
        np.random.seed(seed)
    
    return env


def split_state_from_obs(obs_vec: np.ndarray, num_picker: int) -> tuple[np.ndarray, np.ndarray]:
    """Split observation vector into (environment_state, agent_state).
    
    Args:
        obs_vec: Full observation vector
        num_picker: Number of pickers/grippers
    
    Returns:
        Tuple of (env_state, agent_state) where agent_state contains picker positions
    """
    picker_dims = int(3 * num_picker)
    if obs_vec.ndim != 1 or obs_vec.shape[0] <= picker_dims:
        raise ValueError(f"Unexpected obs vector shape {obs_vec.shape}, cannot split pickers={num_picker}")
    env_state = obs_vec[:-picker_dims].astype(np.float32)
    agent_state = obs_vec[-picker_dims:].astype(np.float32)
    return env_state, agent_state


def evaluate_local_flow_3d_softgym(
    model,
    stats,
    env_name: str = "RopeFlatten",
    env_kwargs: Optional[Dict[str, Any]] = None,
    out_path: str = "vis_softgym.mp4",
    num_envs: int = 16,
    max_steps: int = 300,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    device: Optional[torch.device] = None,
    save_video: bool = True,
    eval_seed: Optional[int] = None,
    num_pickers: int = 2,
    env: Optional[Any] = None,
):
    """
    Evaluate a LocalFlowPolicy3D model on SoftGym environments.
    
    Args:
        model: LocalFlowPolicy3D model in eval() mode
        stats: Dataset statistics for normalization
        env_name: SoftGym environment name (e.g., "RopeFlatten")
        env_kwargs: Optional environment kwargs
        out_path: Path to save video (only saves first env)
        num_envs: Number of parallel environments
        max_steps: Maximum steps per episode
        pred_horizon: Prediction horizon
        obs_horizon: Observation horizon
        action_horizon: Action horizon
        device: Torch device (auto-detected if None)
        save_video: Whether to save video
        eval_seed: Random seed for evaluation
        num_pickers: Number of pickers/grippers
        env: Optional pre-initialized environment. If None, creates a new one.
            If provided, will be reused and reset for each evaluation.
    
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # PyFlex doesn't support true parallelization - all environments share the same PyFlex context
    # So we evaluate sequentially: create ONE environment with num_variations set, then reset with different config_id
    # Follow the pattern from generate_expert_trajs.py:
    # 1. Set num_variations to num_envs in kwargs
    # 2. Create ONE environment (generates num_variations variations at init)
    # 3. Evaluate each variation sequentially by resetting with config_id=i
    
    # Use provided environment or create a new one
    env_provided = env is not None
    if env is None:
        # Set num_variations to exactly num_envs (like in generate_expert_trajs.py)
        if env_kwargs is None:
            env_kwargs = {}
        env_kwargs_with_variations = deepcopy(env_kwargs)
        env_kwargs_with_variations['num_variations'] = num_envs
        
        # Use a single base seed
        if eval_seed is not None:
            base_seed = eval_seed
        else:
            base_seed = 100000
        
        # Create ONE environment with num_variations=num_envs (generates variations at init)
        # This matches generate_expert_trajs.py: create env once, then reset multiple times
        env = build_softgym_env(env_name, env_kwargs_with_variations, seed=base_seed, num_envs=num_envs)
    
    # Evaluate each variation sequentially (PyFlex doesn't support parallelization)
    all_rewards_sum = []
    all_successes = []
    all_max_single_rewards = []
    all_frames = []
    
    pbar = tqdm(total=num_envs, desc="Eval LocalFlowPolicy3D (SoftGym)")
    for env_idx in range(num_envs):
        # Reset environment with config_id=env_idx to use the env_idx-th variation
        try:
            obs_vec = env.reset(config_id=env_idx)
        except TypeError:
            # Some environments might not support config_id parameter
            try:
                obs_vec = env.reset()
            except Exception:
                obs_vec = env.reset(initial_state=None)
        
        # Split observation
        env_state, agent_state = split_state_from_obs(obs_vec, num_pickers)
        obs_full = np.concatenate([env_state, agent_state], axis=0)  # (obs_dim,)
        obs_dim = obs_full.shape[0]
        
        # Rolling observation buffer: (obs_horizon, obs_dim)
        obs_buf = np.repeat(obs_full[None, :], repeats=obs_horizon, axis=0)  # (obs_horizon, obs_dim)
        
        # Episode bookkeeping
        episode_reward = 0.0
        episode_success = False
        max_single_reward = -np.inf
        episode_frames = []
        
        # Try to get initial frame (only for first env)
        if env_idx == 0:
            try:
                initial_frame = env.get_image(128, 128)
                if initial_frame is not None:
                    if initial_frame.dtype != np.uint8:
                        initial_frame = (np.clip(initial_frame, 0, 255)).astype(np.uint8)
                    if initial_frame.ndim == 3 and initial_frame.shape[0] in (1, 3):
                        initial_frame = np.moveaxis(initial_frame, 0, -1)
                    episode_frames.append(initial_frame)
            except Exception:
                pass
        
        action_step_counter = 0
        steps_done = 0
        
        # Run episode
        while steps_done < max_steps:
            # Prepare conditional observation
            nobs = normalize_data(obs_buf[None, :, :], stats=stats["obs"])  # (1, obs_horizon, obs_dim)
            nobs_t = torch.as_tensor(nobs, device=device, dtype=torch.float32)
            
            # Get new action prediction at the start of each action horizon
            if action_step_counter == 0:
                with torch.no_grad():
                    # Get reference position stats (3D picker positions)
                    reference_pos_stats = None
                    if stats and "obs" in stats:
                        picker_start_idx = obs_dim - 3 * num_pickers
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
                    
                    action_dict = model.get_action(
                        obs_seq=nobs_t,
                        action_stats=stats.get("action") if stats else None,
                        reference_pos_stats=reference_pos_stats,
                    )
                    action_pred = action_dict["actions"]  # (1, action_horizon, action_dim) - UNNORMALIZED
                    
                    # Actions are already unnormalized, convert to numpy
                    action_chunk = action_pred.detach().cpu().numpy()[0]  # (action_horizon, action_dim)
            
            # Use the current action from the chunk
            a_t = action_chunk[action_step_counter, :]  # (action_dim,)
            
            # Step environment
            obs_next, rew, done, info = env.step(a_t)
            
            # Update episode metrics
            episode_reward += float(rew)
            max_single_reward = max(max_single_reward, float(rew))
            
            # Check for success
            if isinstance(info, dict) and "is_success" in info:
                episode_success = episode_success or bool(info["is_success"])
            elif hasattr(info, "get"):
                episode_success = episode_success or bool(info.get("is_success", False))
            
            # Update rolling obs buffer
            env_state_next, agent_state_next = split_state_from_obs(obs_next, num_pickers)
            obs_full_next = np.concatenate([env_state_next, agent_state_next], axis=0)
            obs_buf = np.concatenate([obs_buf[1:, :], obs_full_next[None, :]], axis=0)
            
            # Try to render frame (only for first env)
            if env_idx == 0 and steps_done % 5 == 0:
                try:
                    frame = env.get_image(128, 128)
                    if frame is not None:
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, 0, 255)).astype(np.uint8)
                        if frame.ndim == 3 and frame.shape[0] in (1, 3):
                            frame = np.moveaxis(frame, 0, -1)
                        episode_frames.append(frame)
                except Exception:
                    pass
            
            steps_done += 1
            action_step_counter = (action_step_counter + 1) % action_horizon
            
            if done or steps_done >= max_steps:
                break
        
        # Store results for this environment
        all_rewards_sum.append(episode_reward)
        all_successes.append(episode_success)
        all_max_single_rewards.append(max_single_reward)
        if env_idx == 0:
            all_frames = episode_frames
        
        pbar.update(1)
        pbar.set_postfix(env=env_idx, reward=episode_reward, success=episode_success)
    
    pbar.close()
    
    # Close environment only if we created it (not if it was provided)
    if not env_provided:
        try:
            env.close()
        except Exception:
            pass
    
    # Summary
    rewards_sum = np.array(all_rewards_sum)
    successes = np.array(all_successes)
    max_single_reward_per_env = np.array(all_max_single_rewards)
    
    mean_score = float(np.mean(rewards_sum))
    max_score = float(np.max(rewards_sum))
    mean_max_single_reward = float(np.mean(max_single_reward_per_env))
    success_count = int(successes.sum())
    success_rate = success_count / float(num_envs)
    
    print("Mean cumulative score across envs:", mean_score)
    print("Max cumulative score across envs:", max_score)
    print("Mean max single-step reward (avg of max per env):", mean_max_single_reward)
    print(f"Success rate: {success_count}/{num_envs} ({success_rate:.2%})")
    
    # Prepare video
    video_frames = None
    if len(all_frames) > 0:
        video_frames = np.stack(all_frames, axis=0)  # (T, H, W, C)
        
        if save_video:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            iio.imwrite(out_path, video_frames, fps=10)
            print(f"Saved evaluation video to {out_path}")
    
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
