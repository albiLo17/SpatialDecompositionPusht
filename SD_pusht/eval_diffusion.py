import os
import collections
import numpy as np
import torch
from tqdm import tqdm
import time
import zarr
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from diffusers import DDPMScheduler
import imageio.v3 as iio

from gym_pusht.envs import PushTEnv
from SD_pusht.push_t_dataset import PushTStateDataset, normalize_data, unnormalize_data
from network import ConditionalUnet1D


def tile_images(images, nrows=1):
    """
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched : 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        # place column into output
        if batched:
            if is_torch:
                output_image[:, :cur_h, cur_x:next_x, :] = column_image
            else:
                output_image[:, :cur_h, cur_x:next_x, :] = column_image
        else:
            output_image[:cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image


def make_env(render_mode="rgb_array", legacy=True, seed=None):
    def _thunk():
        env = PushTEnv(render_mode=render_mode, legacy=legacy)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk


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
    noise_pred_net,
    stats,
    out_path="vis_all_envs.mp4",
    noise_scheduler=None,
    action_dim=2,
    num_envs=16,
    max_steps=300,
    num_diffusion_iters=100,
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8,
    device=None,
):
    """
    Evaluate a diffusion model on `num_envs` parallel PushTEnv environments and write a tiled video to out_path.

    Args:
      noise_pred_net: a torch.nn.Module already loaded and in eval() mode.
      stats: dataset stats dict required by normalize_data/unnormalize_data.
      out_path: path to write mp4 video.
      num_envs: number of parallel envs.
      max_steps: max environment steps (per-env) to run.
      pred_horizon/obs_horizon/action_horizon: horizons used by the model/dataset.
      num_diffusion_iters: diffusion timesteps used for sampling.
      device: torch.device or None (will infer from model parameters).
    Returns:
      dict with summary: mean_score, max_score, success_rate, steps_done
    """
    if device is None:
        try:
            device = next(noise_pred_net.parameters()).device
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

    # visualization frames: list[T] where each element is list[B] frames
    frames_all = []
    initial_frames = _call_render_all(venv)
    if any(f is not None for f in initial_frames):
        frames_all.append([np.asarray(f, dtype=np.uint8) if f is not None else None for f in initial_frames])

    alive = np.ones((num_envs,), dtype=bool)

    pbar = tqdm(total=max_steps, desc="Eval PushTStateEnv (vectorized)")
    try:
        while steps_done < max_steps:
            B = num_envs

            # prepare conditional observation
            nobs = normalize_data(obs_buf, stats=stats["obs"])
            nobs_t = torch.as_tensor(nobs, device=device, dtype=torch.float32)
            obs_cond = nobs_t.flatten(start_dim=1)

            with torch.no_grad():
                naction = torch.randn((B, pred_horizon, action_dim), device=device)
                noise_scheduler.set_timesteps(num_diffusion_iters)
                for k in noise_scheduler.timesteps:
                    noise_pred = noise_pred_net(sample=naction, timestep=k, global_cond=obs_cond)
                    naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

            action_pred = unnormalize_data(naction.detach().cpu().numpy(), stats=stats["action"])
            start, end = obs_horizon - 1, obs_horizon - 1 + action_horizon
            action_chunk = action_pred[:, start:end, :]  # (B, H, A)

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
    max_score = float(np.max(rewards_sum))
    success_count = int(successes.sum())
    success_rate = success_count / float(num_envs)
    print("Mean score across envs:", mean_score)
    print("Max score across envs:", max_score)
    print(f"Success rate: {success_count}/{num_envs} ({success_rate:.2%})")

    # Save tiled video of all envs
    if len(frames_all) > 0:
        T = len(frames_all)
        B = num_envs

        # Build per-env lists and pad shorter timelines by repeating last frame or a black frame
        per_env_lists = []
        for i in range(B):
            per_env = [frames_all[t][i] for t in range(T) if frames_all[t][i] is not None]
            if len(per_env) == 0:
                H, W = frames_all[0][0].shape[:2]
                per_env = [np.zeros((H, W, 3), dtype=np.uint8)] * T
            elif len(per_env) < T:
                last = per_env[-1]
                per_env.extend([last] * (T - len(per_env)))
            per_env_lists.append(per_env)

        # Reconstruct frames_all_padded as list of T lists of B frames
        frames_all_padded = [[per_env_lists[i][t] for i in range(B)] for t in range(T)]

        # choose tile layout (square-ish)
        nrows = int(np.ceil(np.sqrt(B)))
        tiled_frames = [tile_images(frames_at_t, nrows=nrows) for frames_at_t in frames_all_padded]

        arr = np.stack(tiled_frames, axis=0)  # T, Ht, Wt, 3
        # create directory
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        iio.imwrite(out_path, arr, fps=30)
        print(f"Saved tiled video to {out_path}")

    venv.close()

    return {
        "mean_score": mean_score,
        "max_score": max_score,
        "success_count": success_count,
        "success_rate": success_rate,
        "steps_done": steps_done,
    }


if __name__ == "__main__":
    # ---- Config ----
    dataset_path = "datasets/pusht_cchi_v7_replay.zarr.zip"
    ckpt_path = "log/checkpoints/ema_noise_pred_net_20251017T123038Z.pt"

    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    num_envs = 16
    max_steps = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data stats (normalization) ----
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    stats = dataset.stats

    # ---- Network ----
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim if False else 2,  # keep api compatible
        global_cond_dim=obs_dim if False else (5 * obs_horizon),
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    noise_pred_net.load_state_dict(state_dict)
    noise_pred_net.to(device).eval()
    
    action_dim = 2
    obs_dim = 5
    num_diffusion_iters = 100

    # diffusion scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )


    res = evaluate_model(
        noise_pred_net,
        stats,
        out_path="vis_all_envs.mp4",
        num_envs=num_envs,
        max_steps=max_steps,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        num_diffusion_iters=100,
        device=device,
    )
    print("Evaluation result:", res)
