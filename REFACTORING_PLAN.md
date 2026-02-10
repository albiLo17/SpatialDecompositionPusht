# Multi-Gripper Refactoring Plan

## Overview
Refactor the models and training code to support multiple grippers with:
1. Multiple reference frames (one per gripper)
2. Actions in local frames that are transformed to world frames
3. Action format: [dx1, dy1, dz1, pick1, dx2, dy2, dz2, pick2] for 2 grippers

## Current State
- **Position2DFlowDecoder**: Predicts single 2D reference position (B, 2)
- **LocalFlowPolicy2D**: Takes single reference position, predicts 2D actions (B, act_horizon, 2)
- **Dataset**: Returns reference_pos with shape (B, num_pickers, 2) when use_gripper_segmentation=True

## Target State
- **Position2DFlowDecoder**: Predicts reference positions for all grippers (B, num_pickers, 2)
- **LocalFlowPolicy2D**: 
  - Takes reference positions for all grippers (B, num_pickers, 2)
  - Predicts actions for all grippers (B, act_horizon, act_dim) where act_dim = 4 * num_pickers
  - Actions are in local frames: (dx, dy, dz) relative to each gripper's reference position
  - Transforms local actions to world frame per gripper
- **Training scripts**: Updated to handle multi-gripper format

## Implementation Steps

### Step 1: Update Position2DFlowDecoder
**File**: `SD_pusht/models/local_flow_policy_2d.py`

**Changes**:
1. Add `num_pickers` parameter to `__init__`
2. Change output dimension from 2 to `num_pickers * 2`
3. Update `forward()` to return shape (B, num_pickers, 2) instead of (B, 2)
4. Update `compute_loss()` to handle GT positions with shape (B, num_pickers, 2)
5. Update DirectPositionMLP output layer: `nn.Linear(prev_dim, num_pickers * 2)`
6. Reshape outputs appropriately throughout

**Key changes**:
- `self.POSITION_DIM = 2 * num_pickers` (or keep as 2 and reshape)
- Output reshaping: `position.reshape(B, num_pickers, 2)`
- Loss computation: handle (B, num_pickers, 2) GT positions

### Step 2: Update LocalFlowPolicy2D
**File**: `SD_pusht/models/local_flow_policy_2d.py`

**Changes**:
1. Add `num_pickers` parameter to `__init__`
2. Update `act_dim` handling: expect `act_dim = 4 * num_pickers` (dx, dy, dz, pick_flag per gripper)
3. Update `_transform_obs_to_local_frame()`:
   - Handle multiple reference positions
   - Transform observations relative to each gripper's reference
   - For SoftGym: obs contains [env_state, picker1_pos, picker2_pos, ...]
   - Need to transform picker positions relative to their reference frames
4. Update `get_action()`:
   - Accept `reference_position` with shape (B, num_pickers, 2)
   - Predict actions with shape (B, act_horizon, 4 * num_pickers)
   - Transform local actions to world frame per gripper:
     - For each gripper i: world_action[:, :, i*4:(i+1)*4-1] = local_action[:, :, i*4:(i+1)*4-1] + ref_pos[i]
     - pick_flag (last dim) stays the same
5. Update `compute_loss()`:
   - Accept `reference_position` with shape (B, num_pickers, 2)
   - Transform world actions to local frame per gripper
   - Compute loss on local actions
6. Update FiLM conditioning:
   - Handle multiple reference positions (may need to condition on all or average)
   - Or condition each gripper's action prediction on its own reference

### Step 3: Update Training Scripts
**Files**: 
- `SD_pusht/scripts/train_local_flow_2d.py`
- `SD_pusht/scripts/train_position_decoder_2d.py`
- `SD_pusht/scripts/train_action_policy_2d.py`

**Changes**:
1. Add `--num-pickers` argument (default: 2)
2. Update model initialization:
   - Pass `num_pickers` to Position2DFlowDecoder and LocalFlowPolicy2D
   - Update `act_dim` to `4 * num_pickers`
3. Update dataset loading:
   - Ensure `use_gripper_segmentation=True`
   - Verify `reference_pos` has shape (B, num_pickers, 2)
4. Update loss computation:
   - Handle multi-gripper reference positions
   - Handle multi-gripper actions

### Step 4: Update Dataset
**File**: `SD_pusht/datasets/softgym_segmented_dataset.py`

**Verification**:
- Ensure `__getitem__` returns `reference_pos` with shape (num_pickers, 2) when `use_gripper_segmentation=True`
- This should already be correct based on previous work

### Step 5: Action Transformation Details

**Local Frame Actions**:
- For gripper i: `[dx_i, dy_i, dz_i, pick_flag_i]`
- These are in the local frame relative to gripper i's reference position

**World Frame Transformation**:
- For gripper i:
  - `world_x = ref_pos[i, 0] + dx_i`
  - `world_y = ref_pos[i, 1] + dy_i`  
  - `world_z = ref_pos[i, 2] + dz_i` (if using 3D, otherwise z stays as dz)
  - `pick_flag_i` stays the same

**Note**: For 2D tasks, we might only use (x, y) and ignore z, or z could be predicted directly in world frame.

### Step 6: Observation Transformation

**Current**: For PushT, observations are `[agent_x, agent_y, block_x, block_y, block_angle]`

**For SoftGym**: Observations are `[env_state (30 dims), picker1_x, picker1_y, picker1_z, picker2_x, picker2_y, picker2_z]`

**Local Frame Transformation**:
- For each gripper i, subtract its reference position from its picker position
- Environment state stays the same (or could be transformed relative to average reference)

## Implementation Order

1. **First**: Update Position2DFlowDecoder to predict multiple reference positions
2. **Second**: Update LocalFlowPolicy2D to handle multiple reference positions and predict multi-gripper actions
3. **Third**: Update training scripts
4. **Fourth**: Test and verify

## Testing Checklist

- [ ] Position2DFlowDecoder outputs shape (B, num_pickers, 2)
- [ ] LocalFlowPolicy2D accepts reference_positions with shape (B, num_pickers, 2)
- [ ] LocalFlowPolicy2D predicts actions with shape (B, act_horizon, 4 * num_pickers)
- [ ] Local-to-world transformation works correctly for each gripper
- [ ] World-to-local transformation works correctly for each gripper
- [ ] Training scripts run without errors
- [ ] Loss computation handles multi-gripper format correctly
