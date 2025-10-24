#!/usr/bin/env python3
"""
Convert PushT zarr dataset into a ToyDataset-compatible NPZ used by
spatialdecomposition/SpatialDecomposition/TwoD_table_play/data.py

This builds trajectories of 2D agent positions (x, y) per episode, with
optional fixed-length padding / truncation, and stores them through
ToyDataset.store_to_file so it can be loaded with ToyDataset.from_file.

Example (Non-segmented):
  python SD_pusht/convert_pusht_to_toydataset.py \
    --input datasets/pusht_cchi_v7_replay.zarr.zip \
    --output datasets/pusht_toy_dataset.npz \
    --traj-length 128 --max-episodes 1000

"""

import argparse
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import zarr
import sys


def _open_zarr(path: str):
    """Open zarr roots for 'data' and 'meta' for directory or .zip stores.

    Returns a tuple (data_root, meta_root). If only a single root exists,
    both elements may point to the same handle and expected keys must exist.
    """
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            tmpdir = tempfile.TemporaryDirectory()
            zf.extractall(tmpdir.name)
            data_path = os.path.join(tmpdir.name, "data")
            meta_path = os.path.join(tmpdir.name, "meta")
            if os.path.isdir(data_path) and os.path.isdir(meta_path):
                data_root = zarr.open(data_path, mode="r")
                meta_root = zarr.open(meta_path, mode="r")
            else:
                # Fallback: single root
                entries = [e for e in os.listdir(tmpdir.name) if os.path.isdir(os.path.join(tmpdir.name, e))]
                if not entries:
                    raise RuntimeError("No zarr directory found inside zip archive")
                single_root = zarr.open(os.path.join(tmpdir.name, entries[0]), mode="r")
                data_root = single_root
                meta_root = single_root
            # Keep tmpdir alive by attaching to roots (best-effort)
            data_root._tmpdir = tmpdir  # type: ignore[attr-defined]
            meta_root._tmpdir = tmpdir  # type: ignore[attr-defined]
            return data_root, meta_root
    # plain directory store: try subdirs
    data_path = os.path.join(path, "data")
    meta_path = os.path.join(path, "meta")
    if os.path.isdir(data_path) and os.path.isdir(meta_path):
        return zarr.open(data_path, mode="r"), zarr.open(meta_path, mode="r")
    root = zarr.open(path, mode="r")
    return root, root


def _slice_episodes(episode_ends: np.ndarray, max_episodes: Optional[int]):
    if max_episodes is None:
        return episode_ends
    return episode_ends[:max_episodes]


def _pad_or_truncate(traj: np.ndarray, T: int) -> np.ndarray:
    """Ensure trajectory has length T by tail-padding with last frame or truncating.

    traj: (L, D), returns (T, D)
    """
    L, D = traj.shape
    if L == T:
        return traj
    if L > T:
        return traj[:T]
    out = np.empty((T, D), dtype=traj.dtype)
    out[:L] = traj
    out[L:] = traj[L - 1]
    return out


def segment_trajectory_by_contact(states, actions, contact_threshold=0.1, min_segment_length=5):
    """
    Segment a trajectory based on contact with the object.
    
    Args:
        states: (T, 5) array with [agent_x, agent_y, block_x, block_y, block_angle]
        actions: (T, 2) array with agent actions
        contact_threshold: Minimum movement to consider contact
        min_segment_length: Minimum length for a valid segment
        
    Returns:
        List of (segment_states, segment_actions, contact_flag, start_idx, end_idx)
    """
    T = len(states)
    segments = []
    
    # Calculate block movement (position and angle changes)
    block_pos = states[:, 2:4]  # block_x, block_y
    block_angle = states[:, 4:5]  # block_angle
    
    # Position movement
    block_pos_diff = np.linalg.norm(np.diff(block_pos, axis=0), axis=1)
    # Angle movement  
    block_angle_diff = np.abs(np.diff(block_angle, axis=0).flatten())
    
    # Combined movement metric
    block_movement = np.concatenate([[0], block_pos_diff + block_angle_diff])
    
    # Detect contact periods (when block is moving)
    in_contact = block_movement > contact_threshold
    
    # Find segment boundaries
    segment_starts = []
    segment_ends = []
    contact_flags = []
    
    current_start = 0
    current_contact = in_contact[0]
    
    for i in range(1, T):
        if in_contact[i] != current_contact:
            # Contact state changed, end current segment
            if i - current_start >= min_segment_length:
                segment_starts.append(current_start)
                segment_ends.append(i)
                contact_flags.append(current_contact)
            
            # Start new segment
            current_start = i
            current_contact = in_contact[i]
    
    # Add final segment
    if T - current_start >= min_segment_length:
        segment_starts.append(current_start)
        segment_ends.append(T)
        contact_flags.append(current_contact)
    
    # Create segments
    for start, end, contact in zip(segment_starts, segment_ends, contact_flags):
        segment_states = states[start:end]
        segment_actions = actions[start:end]
        segments.append((segment_states, segment_actions, contact, start, end))
    
    return segments


def load_agent_xy_trajectories(zarr_data_root, zarr_meta_root, traj_length: Optional[int], max_episodes: Optional[int], segment_by_contact: bool = False) -> torch.Tensor:
    """
    Build per-episode trajectories of agent (x,y) positions from PushT zarr.

    Returns a torch.FloatTensor of shape (N, T, 2) if traj_length is provided,
    else (N, L_i, 2) will still be stacked requiring equal lengths; so we
    recommend setting traj_length.
    """
    # data layout in push-t zarr
    # - data/state: (N_total, 5) = [agent_x, agent_y, block_x, block_y, block_angle]
    # - data/action: (N_total, 2)
    # - meta/episode_ends: (num_episodes,) cumulative end indices
    # Try typical layout: data/state and meta/episode_ends
    if "state" in zarr_data_root:
        states = zarr_data_root["state"][:]
    elif "data" in zarr_data_root and "state" in zarr_data_root["data"]:
        states = zarr_data_root["data"]["state"][:]
    else:
        raise KeyError("Cannot find 'state' array in zarr data root")

    # Load actions if available
    if "action" in zarr_data_root:
        actions = zarr_data_root["action"][:]
    elif "data" in zarr_data_root and "action" in zarr_data_root["data"]:
        actions = zarr_data_root["data"]["action"][:]
    else:
        actions = None

    if "episode_ends" in zarr_meta_root:
        episode_ends = zarr_meta_root["episode_ends"][:]
    elif "meta" in zarr_meta_root and "episode_ends" in zarr_meta_root["meta"]:
        episode_ends = zarr_meta_root["meta"]["episode_ends"][:]
    else:
        raise KeyError("Cannot find 'episode_ends' in zarr meta root")

    episode_ends = _slice_episodes(episode_ends, max_episodes)

    trajectories = []
    segment_metadata = []  # Store metadata for each segment
    
    start = 0
    for ep_idx, end in enumerate(episode_ends):
        end = int(end)
        ep_states = states[start:end]
        ep_actions = actions[start:end] if actions is not None else None
        
        if segment_by_contact and ep_actions is not None:
            # Segment the episode based on contact
            segments = segment_trajectory_by_contact(ep_states, ep_actions)
            
            for seg_idx, (seg_states, seg_actions, contact_flag, seg_start, seg_end) in enumerate(segments):
                agent_xy = seg_states[:, :2].astype(np.float32)  # (L, 2)
                if traj_length is not None:
                    agent_xy = _pad_or_truncate(agent_xy, traj_length)
                trajectories.append(agent_xy)
                
                # Store metadata for debugging
                segment_metadata.append({
                    'original_episode': ep_idx,
                    'segment_idx': seg_idx,
                    'contact_flag': contact_flag,
                    'original_start': start + seg_start,
                    'original_end': start + seg_end,
                    'segment_length': len(seg_states)
                })
        else:
            # Original behavior: use full episode
            agent_xy = ep_states[:, :2].astype(np.float32)  # (L, 2)
            if traj_length is not None:
                agent_xy = _pad_or_truncate(agent_xy, traj_length)
            trajectories.append(agent_xy)
            
            segment_metadata.append({
                'original_episode': ep_idx,
                'segment_idx': 0,
                'contact_flag': None,
                'original_start': start,
                'original_end': end,
                'segment_length': len(ep_states)
            })
        
        start = end

    if not trajectories:
        return torch.zeros((0, 0, 2), dtype=torch.float32)

    # If traj_length is None, enforce uniform length if possible
    if traj_length is None:
        lengths = {t.shape[0] for t in trajectories}
        if len(lengths) != 1:
            raise ValueError(
                "Variable-length episodes found; specify --traj-length to pad/truncate to a fixed T"
            )

    stacked = np.stack(trajectories, axis=0)  # (N, T, 2)
    result = torch.from_numpy(stacked)
    
    # Store metadata as attribute for debugging
    if segment_by_contact and len(segment_metadata) > 0:
        # Create a custom tensor with metadata
        class TrajectoryTensor(torch.Tensor):
            def __new__(cls, data, metadata=None):
                obj = super().__new__(cls, data)
                obj._segment_metadata = metadata
                return obj
        
        result = TrajectoryTensor(stacked, segment_metadata)
        print(f"Debug: Created TrajectoryTensor with {len(segment_metadata)} metadata entries")
    
    return result


def main():
    p = argparse.ArgumentParser(description="Convert PushT zarr to ToyDataset NPZ (agent XY trajectories)")
    p.add_argument("--input", required=True, help="Path to PushT zarr dir or .zip")
    p.add_argument("--output", required=True, help="Path to output .npz (or without suffix)")
    p.add_argument("--traj-length", type=int, default=128, help="Fixed trajectory length T (pad/truncate)")
    p.add_argument("--max-episodes", type=int, default=None, help="Max number of episodes to convert")
    p.add_argument("--normalize", action="store_true", help="Normalize XY to [0,1] by dividing by 512")
    p.add_argument("--save-via-toydataset", action="store_true", help="If set, import ToyDataset and save via store_to_file")
    p.add_argument("--segment-by-contact", action="store_true", help="Segment trajectories based on contact with object")
    args = p.parse_args()

    data_root, meta_root = _open_zarr(args.input)

    trajectories = load_agent_xy_trajectories(data_root, meta_root, args.traj_length, args.max_episodes, args.segment_by_contact)

    if args.normalize and trajectories.numel() > 0:
        trajectories = trajectories / 512.0

    # Extract metadata if available
    global_segment_metadata = None
    if args.segment_by_contact and hasattr(trajectories, '_segment_metadata'):
        global_segment_metadata = trajectories._segment_metadata
        contact_segments = sum(1 for m in global_segment_metadata if m['contact_flag'])
        no_contact_segments = sum(1 for m in global_segment_metadata if not m['contact_flag'])
        print(f"Segmentation results:")
        print(f"  - Total segments: {len(global_segment_metadata)}")
        print(f"  - Contact segments: {contact_segments}")
        print(f"  - No-contact segments: {no_contact_segments}")
        print(f"  - Average segment length: {np.mean([m['segment_length'] for m in global_segment_metadata]):.1f}")
    elif args.segment_by_contact:
        print("Warning: Segmentation enabled but no metadata found!")

    out_path = Path(args.output)
    if args.save_via_toydataset:
        # Save using ToyDataset API
        from SpatialDecomposition.TwoD_table_play.data import ToyDataset
        ds = ToyDataset(
            trajectories=trajectories,
            trajectory_labels_numerical=None,
            label_list=None,
        )
        ds.store_to_file(out_path)
        out_file = out_path.with_suffix(".npz")
        N = len(ds)
        T = trajectories.shape[1] if trajectories.ndim == 3 and N > 0 else 0
        print(f"Saved via ToyDataset.store_to_file: N={N}, T={T}, D=2 -> {out_file}")
        
        # Save segmentation metadata separately if available
        if args.segment_by_contact and global_segment_metadata is not None:
            metadata_file = out_path.with_suffix("_metadata.npz")
            np.savez(metadata_file, 
                    segment_metadata=global_segment_metadata,
                    contact_threshold=0.1,
                    min_segment_length=5)
            print(f"Saved segmentation metadata: {metadata_file}")
    else:
        # Persist using the same keys as ToyDataset.store_to_file, without importing it
        # Keys: class_type, trajectories, labels, label_list
        out_file = out_path.with_suffix(".npz")
        # Store a hint for the class to avoid pickling heavy dependencies
        class_type_module = "SpatialDecomposition.TwoD_table_play.data"
        class_type_name = "ToyDataset"
        np.savez(
            out_file,
            class_type_module=class_type_module,
            class_type_name=class_type_name,
            trajectories=trajectories.numpy(),
            labels=None,
            label_list=None,
        )

        N = trajectories.shape[0] if trajectories.ndim == 3 else 0
        T = trajectories.shape[1] if trajectories.ndim == 3 and N > 0 else 0
        print(f"Saved ToyDataset-compatible NPZ: N={N}, T={T}, D=2 -> {out_file}")


if __name__ == "__main__":
    main()


