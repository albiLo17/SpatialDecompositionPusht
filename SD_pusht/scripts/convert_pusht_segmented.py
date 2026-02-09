#!/usr/bin/env python3
"""
Simplified converter that creates segmented datasets with metadata.


Example (Segmented by contact):
  python SD_pusht/convert_pusht_segmented.py \
    --input datasets/pusht_cchi_v7_replay.zarr.zip \
    --output datasets/pusht_toy_dataset_segmented_normalized.npz \
    --traj-length 64 --normalize
    
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

# Add paths for imports
REPO_ROOT = "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht"
SPATIALDECOMP_PATH = os.path.join(REPO_ROOT, "spatialdecomposition")
if SPATIALDECOMP_PATH not in sys.path:
    sys.path.append(SPATIALDECOMP_PATH)

from SpatialDecomposition.TwoD_table_play.data import ToyDataset

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    min_val = torch.min(data, axis=0).values
    max_val = torch.max(data, axis=0).values
    
    return min_val, max_val

def segment_trajectory_by_contact(states, actions, contact_threshold=0.1, min_segment_length=5):
    """Segment a trajectory based on contact with the object."""
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


def main():
    parser = argparse.ArgumentParser(description="Convert PushT zarr to segmented ToyDataset")
    parser.add_argument("--input", required=True, help="Path to PushT zarr dir or .zip")
    parser.add_argument("--output", required=True, help="Path to output .npz")
    parser.add_argument("--traj-length", type=int, default=64, help="Fixed trajectory length T")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max number of episodes")
    parser.add_argument("--normalize", action="store_true", help="Normalize XY to [0,1]")
    args = parser.parse_args()

    # Load zarr data
    if args.input.endswith(".zip"):
        with zipfile.ZipFile(args.input, "r") as zf:
            tmpdir = tempfile.TemporaryDirectory()
            zf.extractall(tmpdir.name)
            data_root = zarr.open(os.path.join(tmpdir.name, "data"), mode="r")
            meta_root = zarr.open(os.path.join(tmpdir.name, "meta"), mode="r")
    else:
        data_root = zarr.open(os.path.join(args.input, "data"), mode="r")
        meta_root = zarr.open(os.path.join(args.input, "meta"), mode="r")

    states = data_root["state"][:]
    actions = data_root["action"][:]
    episode_ends = meta_root["episode_ends"][:]

    if args.max_episodes:
        episode_ends = episode_ends[:args.max_episodes]

    # Process episodes with segmentation
    all_trajectories = []
    all_metadata = []
    
    start = 0
    for ep_idx, end in enumerate(episode_ends):
        end = int(end)
        ep_states = states[start:end]
        ep_actions = actions[start:end]
        
        # Segment the episode
        segments = segment_trajectory_by_contact(ep_states, ep_actions)
        
        for seg_idx, (seg_states, seg_actions, contact_flag, seg_start, seg_end) in enumerate(segments):
            # Extract agent XY and pad/truncate
            agent_xy = seg_states[:, :2].astype(np.float32)
            
            # Pad or truncate to fixed length
            if len(agent_xy) > args.traj_length:
                agent_xy = agent_xy[:args.traj_length]
            else:
                # Pad with last position
                padded = np.zeros((args.traj_length, 2), dtype=np.float32)
                padded[:len(agent_xy)] = agent_xy
                if len(agent_xy) > 0:
                    padded[len(agent_xy):] = agent_xy[-1]
                agent_xy = padded
            
            all_trajectories.append(agent_xy)
            
            # Store metadata
            all_metadata.append({
                'original_episode': ep_idx,
                'segment_idx': seg_idx,
                'contact_flag': contact_flag,
                'original_start': start + seg_start,
                'original_end': start + seg_end,
                'segment_length': len(seg_states)
            })
        
        start = end

    # Convert to tensors
    trajectories = torch.from_numpy(np.stack(all_trajectories, axis=0))
    
    # implement proper normalization between [-1, 1] using the min and max of the data
    
    if args.normalize:
        min_val, max_val = get_data_stats(trajectories)
        trajectories = (trajectories - min_val) / (max_val - min_val)
        trajectories = trajectories * 2 - 1

    # Print statistics
    contact_segments = sum(1 for m in all_metadata if m['contact_flag'])
    no_contact_segments = sum(1 for m in all_metadata if not m['contact_flag'])
    print(f"Segmentation results:")
    print(f"  - Total segments: {len(all_metadata)}")
    print(f"  - Contact segments: {contact_segments}")
    print(f"  - No-contact segments: {no_contact_segments}")
    print(f"  - Average segment length: {np.mean([m['segment_length'] for m in all_metadata]):.1f}")

    # Save dataset
    dataset = ToyDataset(
        trajectories=trajectories,
        trajectory_labels_numerical=None,
        label_list=None,
    )
    
    out_path = Path(args.output)
    dataset.store_to_file(out_path)
    print(f"Saved dataset: {out_path.with_suffix('.npz')}")
    
    # Save metadata, storing the min and max of the data
    metadata_path = out_path.parent / f"{out_path.stem}_metadata.npz"
    if args.normalize:
        np.savez(metadata_path, 
            segment_metadata=all_metadata,
            contact_threshold=0.1,
            min_segment_length=5,
            min_val=min_val.numpy(),
            max_val=max_val.numpy())
    else:
        np.savez(metadata_path, 
            segment_metadata=all_metadata,
            contact_threshold=0.1,
            min_segment_length=5)
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
