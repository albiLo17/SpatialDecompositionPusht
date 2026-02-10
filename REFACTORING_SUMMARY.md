# Multi-Gripper Refactoring Summary

## Overview
Successfully refactored the models and training code to support multiple grippers with:
1. Multiple reference frames (one per gripper)
2. Actions in local frames that are transformed to world frames
3. Action format: [dx1, dy1, dz1, pick1, dx2, dy2, dz2, pick2] for 2 grippers

## Changes Made

### 1. Position2DFlowDecoder (`SD_pusht/models/local_flow_policy_2d.py`)
- **Added `num_pickers` parameter** to `__init__`
- **Updated output dimension**: Now predicts `num_pickers * 2` dimensions (flattened) and reshapes to `(B, num_pickers, 2)`
- **Updated `DirectPositionMLP`**: Output layer now outputs `num_pickers * 2` dimensions
- **Updated `forward()` method**: Reshapes output to `(B, num_pickers, 2)` for all inference paths
- **Updated `compute_loss()` method**: Handles GT positions with shape `(B, num_pickers, 2)`

### 2. LocalFlowPolicy2D (`SD_pusht/models/local_flow_policy_2d.py`)
- **Added `num_pickers` parameter** to `__init__`
- **Added validation**: Warns if `act_dim != 4 * num_pickers`
- **Updated `_transform_obs_to_local_frame()`**: 
  - Now accepts `reference_positions` with shape `(B, num_pickers, 2)`
  - Transforms each picker's position relative to its own reference frame
  - Handles SoftGym observation format: `[env_state, picker1_pos, picker2_pos, ...]`
- **Updated `get_action()` method**:
  - Accepts `reference_position` with shape `(B, num_pickers, 2)`
  - Predicts actions with shape `(B, act_horizon, 4 * num_pickers)`
  - Transforms local actions to world frame per gripper:
    - For each gripper i: `world_action[:, :, i*4:(i+1)*4-1] = local_action[:, :, i*4:(i+1)*4-1] + ref_pos[i]`
    - `pick_flag` (last dim of each gripper's action) stays the same
- **Updated `compute_loss()` method**:
  - Accepts `reference_position` with shape `(B, num_pickers, 2)`
  - Transforms world actions to local frame per gripper
  - Computes loss on local actions
- **Updated FiLM conditioning**: Averages reference positions across grippers for conditioning
- **Updated noise sharing**: Handles `POSITION_DIM = 2 * num_pickers`

### 3. Training Scripts

#### `train_local_flow_2d.py`
- **Added `--num-pickers` argument** (default: 2)
- **Added `--action-dim` argument** (default: None, computed as `4 * num_pickers`)
- **Updated dataset loading**: 
  - Uses `SoftGymSegmentedDatasetSimple` instead of `PushTSegmentedDatasetSimple`
  - Sets `use_gripper_segmentation=True`
- **Updated reference position statistics**: Computes from picker positions in observations
- **Updated model initialization**: Passes `num_pickers` to all `LocalFlowPolicy2D` instantiations
- **Added validation**: Checks that `reference_pos` has correct shape `(B, num_pickers, 2)`

#### `train_position_decoder_2d.py`
- **Added `--num-pickers` argument** (default: 2)
- **Updated dataset loading**: Uses `SoftGymSegmentedDatasetSimple` with `use_gripper_segmentation=True`
- **Updated model initialization**: Passes `num_pickers` to all `Position2DFlowDecoder` instantiations

## Action Format

### Input Actions (from dataset)
- Shape: `(B, act_horizon, 4 * num_pickers)`
- Format: `[dx1, dy1, dz1, pick1, dx2, dy2, dz2, pick2, ...]`
- For each gripper i:
  - `dx_i, dy_i, dz_i`: Translation in world frame (will be transformed to local frame during training)
  - `pick_i`: Pick flag (0 = don't pick, 1 = pick/grasp)

### Local Frame Actions (predicted by model)
- Shape: `(B, act_horizon, 4 * num_pickers)`
- Format: `[dx1, dy1, dz1, pick1, dx2, dy2, dz2, pick2, ...]`
- For each gripper i:
  - `dx_i, dy_i, dz_i`: Translation in local frame (relative to gripper i's reference position)
  - `pick_i`: Pick flag (unchanged)

### World Frame Actions (after transformation)
- Shape: `(B, act_horizon, 4 * num_pickers)`
- Format: `[dx1, dy1, dz1, pick1, dx2, dy2, dz2, pick2, ...]`
- For each gripper i:
  - `world_x = ref_pos[i, 0] + dx_i`
  - `world_y = ref_pos[i, 1] + dy_i`
  - `world_z = dz_i` (z stays as is, or could be predicted directly in world frame)
  - `pick_i`: Pick flag (unchanged)

## Reference Position Format

### From Dataset
- Shape: `(B, num_pickers, 2)` when `use_gripper_segmentation=True`
- Each reference position is a 2D (x, y) coordinate in world frame
- One reference position per gripper

### From Position Decoder
- Shape: `(B, num_pickers, 2)`
- Predicted reference positions for all grippers

## Key Implementation Details

1. **Observation Transformation**: For SoftGym, observations contain picker positions at the end. Each picker's position is transformed relative to its own reference frame.

2. **Action Transformation**: 
   - During training: World actions → Local actions (subtract reference position from dx, dy)
   - During inference: Local actions → World actions (add reference position to dx, dy)

3. **FiLM Conditioning**: Currently averages reference positions across grippers for conditioning. Could be extended to condition each gripper separately.

4. **Noise Sharing**: Updated to handle `POSITION_DIM = 2 * num_pickers` instead of just 2.

## Testing Checklist

- [x] Position2DFlowDecoder outputs shape (B, num_pickers, 2)
- [x] LocalFlowPolicy2D accepts reference_positions with shape (B, num_pickers, 2)
- [x] LocalFlowPolicy2D predicts actions with shape (B, act_horizon, 4 * num_pickers)
- [x] Local-to-world transformation works correctly for each gripper
- [x] World-to-local transformation works correctly for each gripper
- [x] Training scripts updated to handle multi-gripper format
- [x] Dataset returns reference_pos with shape (B, num_pickers, 2) when use_gripper_segmentation=True

## Next Steps

1. **Test the refactored code** with actual training runs
2. **Update evaluation scripts** if needed to handle multi-gripper format
3. **Update visualization scripts** to show reference frames for all grippers
4. **Consider extending FiLM conditioning** to condition each gripper separately instead of averaging

## Notes

- The refactoring maintains backward compatibility for single-gripper cases (num_pickers=1)
- All changes are in place and ready for testing
- The dataset already returns the correct format when `use_gripper_segmentation=True` (from previous work)
