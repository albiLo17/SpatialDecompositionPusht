#!/usr/bin/env python3
"""Replay demonstrations from a zarr dataset."""

import time
import argparse
import zarr
import numpy as np
from gym_pusht.envs.pusht import PushTEnv
import tqdm

from SD_pusht.utils.environment import apply_legacy_state
from SD_pusht.utils.normalization import unnormalize_data


def unnormalize(ndata, stats):
    """Invert the dataset normalize_data -> raw = ( (ndata+1)/2 )*(max-min) + min"""
    return unnormalize_data(ndata, stats)


def main(dataset_path, episode_idx=0, speed=1.0, normalized_actions=False, headless=False):
    """Replay a single episode from the dataset.
    
    Args:
        dataset_path: Path to zarr dataset.
        episode_idx: Index of episode to replay.
        speed: Playback speed multiplier (1.0 = real time).
        normalized_actions: Whether actions in dataset are normalized.
        headless: Run without rendering window.
    """
    ds = zarr.open(dataset_path, 'r')
    episode_ends = ds['meta']['episode_ends'][:]
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise ValueError(f"episode_idx must be in [0, {len(episode_ends)-1}]")

    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    actions = ds['data']['action'][start:end]
    states = ds['data']['state'][start:end]

    # If actions were stored normalized in [-1,1], unnormalize using stats
    if normalized_actions:
        stats = ds.attrs.get('stats', None)
        # fallback: try to read meta saved stats under root (some datasets may not include)
        if stats is None and 'stats' in ds:
            stats = ds['stats'][:]
        if stats is None:
            raise RuntimeError("Dataset does not contain stored stats; provide raw actions or stats.")
        # expect stats structure similar to push_t_dataset.PushTStateDataset.stats
        # here we assume stats['action'] with 'min'/'max' arrays
        actions = unnormalize(actions, stats['action'])

    render_mode = None if headless else "human"
    env = PushTEnv(render_mode=render_mode, legacy=True)
    # ensure environment will open human window and reset to the episode's first state
    options = {"reset_to_state": states[0]}
    obs, info = env.reset(options=options)
    
    # use the env metadata key used in PushTEnv for fps
    dt = 1.0 / env.metadata.get("render_fps", 10)
    delay = dt / max(1.0, speed)
    
    print(f"Playing back episode {episode_idx} with {len(actions)} steps at speed x{speed:.2f} {'(headless)' if headless else ''}...")

    for a in tqdm.tqdm(actions):
        # ensure action is numpy float array of shape (2,)
        action = np.array(a, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if not headless:
            env.render()
        time.sleep(delay)
        if terminated or truncated:
            break
        
    print("Episode playback finished.")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay PushT demonstrations")
    ap.add_argument("--dataset", default="datasets/pusht_cchi_v7_replay.zarr.zip", 
                   help="path to zarr dataset (e.g. pusht_cchi_v7_replay.zarr.zip)")
    ap.add_argument("--episode", "-e", type=int, default=0,
                   help="Episode index to replay")
    ap.add_argument("--speed", "-s", type=float, default=1.0,
                   help="playback speed multiplier (1.0 = real time)")
    ap.add_argument("--normalized-actions", action="store_true",
                   help="set if actions in dataset are normalized [-1,1]")
    ap.add_argument("--headless", action="store_true",
                   help="run without opening a rendering window (headless)")
    args = ap.parse_args()
    main(args.dataset, episode_idx=args.episode, speed=args.speed, 
         normalized_actions=args.normalized_actions, headless=args.headless)

