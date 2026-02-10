# How the Segmented Dataset Works

## Overview

The `SoftGymSegmentedDatasetSimple` class segments trajectories based on gripper grasp/release events and assigns reference frames to each training sample. This document explains the process step by step.

## Step 1: Loading Data

When you create a dataset instance:
```python
dataset = SoftGymSegmentedDatasetSimple(
    dataset_path="datasets/ropeflatten_expert.zarr",
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8,
    demo_indices=[0],  # Use episode 0
    use_gripper_segmentation=True,
    min_segment_length=5,
)
```

The dataset:
1. Loads raw data from the zarr file (states, actions, episode boundaries)
2. Extracts the specified episodes (if `demo_indices` is provided)
3. Normalizes the data using dataset statistics

## Step 2: Segmentation

The dataset segments each episode based on **gripper grasp/release events**:

### How Gripper Segmentation Works

For each gripper independently:
1. **Extract pick flags**: From actions, extract the `pick_flag` for each gripper
   - Actions are: `[dx1, dy1, dz1, pick_flag1, dx2, dy2, dz2, pick_flag2, ...]`
   - `pick_flag > 0.5` means "grasping", `pick_flag <= 0.5` means "releasing"

2. **Find transitions**: Detect when a gripper transitions between grasp and release
   - A new segment starts at each transition point
   - Example: If gripper goes from releasing → grasping → releasing, you get 3 segments

3. **Create segments**: Each segment is a `(start_frame, end_frame)` tuple
   - Segments must be at least `min_segment_length` frames long
   - If no valid segments found, the whole episode is treated as one segment

### Example

For a gripper with actions:
```
Frame 0-10:  pick_flag = 0.0 (releasing)
Frame 11-30: pick_flag = 1.0 (grasping)  ← Transition at frame 11
Frame 31-50: pick_flag = 0.0 (releasing)  ← Transition at frame 31
```

This creates segments:
- Segment 0: frames 0-10 (releasing)
- Segment 1: frames 11-30 (grasping)
- Segment 2: frames 31-50 (releasing)

The result is stored in `dataset.gripper_segment_bounds`:
```python
{
    "demo_0": {
        0: [(0, 10), (11, 30), (31, 50)],  # Gripper 0 segments
        1: [(0, 15), (16, 45)],            # Gripper 1 segments (different!)
    }
}
```

## Step 3: Creating Training Samples

The dataset creates training samples using `create_sample_indices()`:
- Each sample has: `(buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)`
- Samples are created with proper padding for `obs_horizon` and `action_horizon`

## Step 4: Assigning Reference Frames

For each training sample, the dataset assigns a **reference frame** for each gripper:

### Reference Frame Logic

1. **Find which segment the sample belongs to**:
   - Look at `buffer_start_idx` (the start of the observation window)
   - Find which gripper segment contains this frame

2. **Determine reference frame**:
   - **If there's a next segment**: Reference = start of next segment
   - **If it's the last segment**: Reference = end of episode

### Example

For gripper 0 with segments `[(0, 10), (11, 30), (31, 50)]`:
- Sample at frame 5 (in segment 0): Reference = frame 11 (start of segment 1)
- Sample at frame 20 (in segment 1): Reference = frame 31 (start of segment 2)
- Sample at frame 40 (in segment 2, last): Reference = frame 49 (end of episode)

The reference frame map is stored in `dataset.reference_frame_maps[gripper_id]`:
- An array where `reference_frame_maps[gripper_id][sample_idx]` gives the reference frame index

## Step 5: Getting a Sample

When you call `dataset[idx]`:

1. **Get observation and action sequences**:
   - Extract normalized observation window (last `obs_horizon` frames)
   - Extract action sequence (next `action_horizon` frames)

2. **Get reference position(s)**:
   - For each gripper, look up the reference frame from `reference_frame_maps[gripper_id][idx]`
   - Extract the gripper's (x, y) position at that reference frame
   - Return as `reference_pos` with shape `(num_pickers, 2)`

## How Visualization Uses This

The visualization script:

1. **Creates a dataset instance** (same as training):
   ```python
   dataset = SoftGymSegmentedDatasetSimple(
       dataset_path=dataset_path,
       demo_indices=[episode_idx],
       use_gripper_segmentation=True,
   )
   ```

2. **Gets segmentation information**:
   ```python
   demo_id = f"demo_{0}"  # Since only one episode loaded, it's demo_0
   gripper_segment_bounds = dataset.gripper_segment_bounds.get(demo_id, {})
   ```

3. **Computes reference frames for visualization**:
   - For each frame in the episode:
     - Find which segment it belongs to (for each gripper)
     - Compute reference frame using the same logic as the dataset
     - Get the 3D position at that reference frame
   - This matches exactly what the dataset does during training!

## Key Points

- **Segmentation is per-gripper**: Each gripper has its own segments based on its own grasp/release events
- **Reference frames change**: As you move through segments, the reference frame updates to point to the next segment's start
- **Last segment special case**: If a gripper doesn't change state after its last segment, the reference is the end of the episode
- **Visualization replicates training logic**: The visualization uses the same segmentation and reference frame computation as the dataset

## Why Reference Frames Matter

The reference frame tells the policy: "At the end of this segment, the gripper should be at this position." This allows the policy to learn to:
- Move the gripper toward the next segment's starting position
- Plan ahead for the next action phase (grasp or release)
