"""SoftGym state dataset with segmentation support for local flow policy training.

This dataset extends the zarr-based dataset format to:
1. Segment trajectories using environment state movement-based segmentation
2. Extract reference positions (last point of each segment)
3. Handle cross-segment padding when predictions extend into next segment
"""

import numpy as np
import torch
import zarr
import os
import sys
from typing import Optional, Dict, List, Tuple

from SD_pusht.utils.normalization import get_data_stats, normalize_data
from SD_pusht.datasets.push_t_dataset import (
    create_sample_indices,
    sample_sequence,
    PushTStateDataset,
)


def segment_trajectory_by_gripper_events(
    actions: np.ndarray,
    num_pickers: int = 2,
    min_segment_length: int = 5
) -> Dict[int, List[Tuple[int, int]]]:
    """Segment a trajectory based on gripper grasp/release events for each gripper.
    
    Args:
        actions: Action array, shape (T, act_dim). Actions are [dx, dy, dz, pick_flag] * num_pickers
        num_pickers: Number of grippers/pickers
        min_segment_length: Minimum length for a valid segment
    
    Returns:
        Dictionary mapping gripper_id to list of (start, end) tuples for each segment.
        Segments are defined by grasp/release events: a new segment starts when a gripper
        transitions from release to grasp or vice versa.
    """
    T = len(actions)
    gripper_segments = {i: [] for i in range(num_pickers)}
    
    if T < min_segment_length:
        # Episode too short, return single segment for each gripper
        for i in range(num_pickers):
            gripper_segments[i] = [(0, T)]
        return gripper_segments
    
    # Extract pick flags for each gripper
    # Actions are [dx1, dy1, dz1, pick_flag1, dx2, dy2, dz2, pick_flag2, ...]
    pick_flags = {}  # gripper_id -> array of pick flags
    for gripper_id in range(num_pickers):
        pick_flag_idx = 3 + gripper_id * 4  # 4th element of each gripper's action
        pick_flags[gripper_id] = actions[:, pick_flag_idx] > 0.5  # True = grasp, False = release
    
    # Segment each gripper independently
    for gripper_id in range(num_pickers):
        flags = pick_flags[gripper_id]
        segments = []
        
        # Find transitions (grasp <-> release)
        current_start = 0
        current_state = flags[0]  # True = grasping, False = releasing
        
        for i in range(1, T):
            if flags[i] != current_state:
                # State changed (grasp <-> release), end current segment
                if i - current_start >= min_segment_length:
                    segments.append((current_start, i))
                
                # Start new segment
                current_start = i
                current_state = flags[i]
        
        # Add final segment
        if T - current_start >= min_segment_length:
            segments.append((current_start, T))
        
        # If no segments found, treat whole episode as one segment
        if len(segments) == 0:
            segments = [(0, T)]
        
        gripper_segments[gripper_id] = segments
    
    return gripper_segments


def segment_trajectory_by_env_movement(
    states: np.ndarray,
    actions: np.ndarray,
    env_state_start_idx: int,
    contact_threshold: float = 0.1,
    min_segment_length: int = 5
) -> List[Tuple[int, int]]:
    """Segment a trajectory based on environment state movement.
    
    Args:
        states: State array, shape (T, state_dim). Assumes states contain:
            - states[:, :agent_state_dim]: agent position/state
            - states[:, agent_state_dim:]: environment state (object configuration)
        actions: Action array, shape (T, act_dim)
        env_state_start_idx: Index where environment state starts in the state array
        contact_threshold: Threshold for detecting object movement
        min_segment_length: Minimum length for a valid segment
    
    Returns:
        List of (start, end) tuples for each segment.
    """
    T = len(states)
    segments = []
    
    if T < min_segment_length:
        # Episode too short, return as single segment
        return [(0, T)]
    
    # Extract environment state (object configuration)
    env_states = states[:, env_state_start_idx:]
    
    # Calculate environment state movement (change in object configuration)
    # Use L2 norm of differences
    if env_states.shape[0] > 1:
        env_state_diff = np.linalg.norm(np.diff(env_states, axis=0), axis=1)
        # Add zero at the beginning to match length
        env_movement = np.concatenate([[0], env_state_diff])
    else:
        env_movement = np.array([0])
    
    # Detect movement periods (when object is moving)
    in_movement = env_movement > contact_threshold
    
    # Find segment boundaries
    segment_starts = []
    segment_ends = []
    
    current_start = 0
    current_movement = in_movement[0]
    
    for i in range(1, T):
        if in_movement[i] != current_movement:
            # Movement state changed, end current segment
            if i - current_start >= min_segment_length:
                segment_starts.append(current_start)
                segment_ends.append(i)
            
            # Start new segment
            current_start = i
            current_movement = in_movement[i]
    
    # Add final segment
    if T - current_start >= min_segment_length:
        segment_starts.append(current_start)
        segment_ends.append(T)
    
    # Create segment bounds list
    for start, end in zip(segment_starts, segment_ends):
        segments.append((start, end))
    
    # If no segments found, treat whole episode as one segment
    if len(segments) == 0:
        segments = [(0, T)]
    
    return segments


class SoftGymSegmentedDatasetSimple(PushTStateDataset):
    """Simplified dataset for SoftGym with segmentation support.
    
    This class uses the same simple dataset creation as PushTStateDataset,
    but adds segment-based frame assignment. Each sample gets a reference
    frame based on which segment it belongs to.
    """
    
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        max_demos: Optional[int] = None,
        demo_indices: Optional[List[int]] = None,
        use_contact_segmentation: bool = True,
        use_gripper_segmentation: bool = True,
        contact_threshold: float = 0.1,
        min_segment_length: int = 5,
        segment_bounds: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        gripper_segment_bounds: Optional[Dict[str, Dict[int, List[Tuple[int, int]]]]] = None,
    ):
        """Initialize simplified SoftGym segmented dataset.
        
        Args:
            dataset_path: Path to zarr dataset file.
            pred_horizon: Number of steps to predict ahead.
            obs_horizon: Number of observation steps to use.
            action_horizon: Number of action steps to execute.
            max_demos: Maximum number of episodes to use (None = all). Ignored if demo_indices is provided.
            demo_indices: Optional list of episode indices to use (e.g., [0, 1, 5, 10]). 
                         If provided, only these episodes will be used.
            use_contact_segmentation: Whether to use movement-based segmentation (legacy).
            use_gripper_segmentation: Whether to use gripper grasp/release segmentation (default: True).
            contact_threshold: Threshold for detecting object movement (if using contact segmentation).
            min_segment_length: Minimum length for a valid segment.
            segment_bounds: Optional pre-computed segment bounds. If None, will compute from movement.
            gripper_segment_bounds: Optional pre-computed gripper segment bounds. 
                                   Dict mapping demo_id to dict mapping gripper_id to segments.
        """
        # First, load the basic dataset structure
        dataset_root = zarr.open(dataset_path, 'r')
        all_train_data = {
            'action': dataset_root['data']['action'][:],
            'obs': dataset_root['data']['state'][:]
        }
        all_episode_ends = dataset_root['meta']['episode_ends'][:]
        
        # Get metadata
        meta = dataset_root['meta']
        self.agent_state_dim = int(meta.attrs.get('agent_state_dim', 0))
        self.env_state_dim = int(meta.attrs.get('env_state_dim', 0))
        self.env_state_start_idx = self.agent_state_dim  # Environment state starts after agent state
        self.num_pickers = int(meta.attrs.get('num_picker', 2))  # Default to 2 pickers
        
        # Determine which episodes to use
        if demo_indices is not None:
            # Use specified demo indices
            demo_indices = sorted(demo_indices)  # Sort to maintain order
            # Validate indices
            max_idx = len(all_episode_ends)
            demo_indices = [idx for idx in demo_indices if 0 <= idx < max_idx]
            if len(demo_indices) == 0:
                raise ValueError(f"No valid demo indices provided. Valid range: 0-{max_idx-1}")
            
            # Extract episode data for selected demos
            episode_ends = []
            train_data = {'action': [], 'obs': []}
            current_offset = 0
            
            for demo_idx in demo_indices:
                # Get episode boundaries
                start_idx = all_episode_ends[demo_idx - 1] if demo_idx > 0 else 0
                end_idx = all_episode_ends[demo_idx]
                
                # Extract episode data
                episode_data_action = all_train_data['action'][start_idx:end_idx]
                episode_data_obs = all_train_data['obs'][start_idx:end_idx]
                
                # Filter out final stationary states
                episode_data_action, episode_data_obs = self._filter_episode_stationary_final_states(
                    episode_data_action, episode_data_obs, action_threshold=1e-4
                )
                
                train_data['action'].append(episode_data_action)
                train_data['obs'].append(episode_data_obs)
                
                # Update episode end index (relative to new concatenated data)
                current_offset += len(episode_data_action)
                episode_ends.append(current_offset)
            
            # Concatenate all episodes
            train_data['action'] = np.concatenate(train_data['action'], axis=0)
            train_data['obs'] = np.concatenate(train_data['obs'], axis=0)
            episode_ends = np.array(episode_ends)
            
            # Store demo indices used
            self.demo_indices = demo_indices
        elif max_demos is not None:
            # Use first max_demos episodes
            episode_ends = all_episode_ends[:max_demos]
            # Extract and filter each episode
            train_data = {'action': [], 'obs': []}
            filtered_episode_ends = []
            current_offset = 0
            
            for demo_idx in range(max_demos):
                start_idx = all_episode_ends[demo_idx - 1] if demo_idx > 0 else 0
                end_idx = all_episode_ends[demo_idx]
                
                # Extract episode data
                episode_data_action = all_train_data['action'][start_idx:end_idx]
                episode_data_obs = all_train_data['obs'][start_idx:end_idx]
                
                # Filter out final stationary states
                episode_data_action, episode_data_obs = self._filter_episode_stationary_final_states(
                    episode_data_action, episode_data_obs, action_threshold=1e-4
                )
                
                train_data['action'].append(episode_data_action)
                train_data['obs'].append(episode_data_obs)
                
                # Update episode end index (relative to new concatenated data)
                current_offset += len(episode_data_action)
                filtered_episode_ends.append(current_offset)
            
            # Concatenate all episodes
            train_data['action'] = np.concatenate(train_data['action'], axis=0)
            train_data['obs'] = np.concatenate(train_data['obs'], axis=0)
            episode_ends = np.array(filtered_episode_ends)
            # Store demo indices used (0 to max_demos-1)
            self.demo_indices = list(range(max_demos))
        else:
            # Use all episodes - filter each one
            train_data = {'action': [], 'obs': []}
            filtered_episode_ends = []
            current_offset = 0
            
            for demo_idx in range(len(all_episode_ends)):
                start_idx = all_episode_ends[demo_idx - 1] if demo_idx > 0 else 0
                end_idx = all_episode_ends[demo_idx]
                
                # Extract episode data
                episode_data_action = all_train_data['action'][start_idx:end_idx]
                episode_data_obs = all_train_data['obs'][start_idx:end_idx]
                
                # Filter out final stationary states
                episode_data_action, episode_data_obs = self._filter_episode_stationary_final_states(
                    episode_data_action, episode_data_obs, action_threshold=1e-4
                )
                
                train_data['action'].append(episode_data_action)
                train_data['obs'].append(episode_data_obs)
                
                # Update episode end index (relative to new concatenated data)
                current_offset += len(episode_data_action)
                filtered_episode_ends.append(current_offset)
            
            # Concatenate all episodes
            train_data['action'] = np.concatenate(train_data['action'], axis=0)
            train_data['obs'] = np.concatenate(train_data['obs'], axis=0)
            episode_ends = np.array(filtered_episode_ends)
            # Store demo indices used (all)
            self.demo_indices = list(range(len(all_episode_ends)))
        
        # Store segmentation parameters
        self.contact_threshold = contact_threshold
        self.min_segment_length = min_segment_length
        self.use_gripper_segmentation = use_gripper_segmentation
        
        # Compute gripper segment bounds if using gripper segmentation
        if use_gripper_segmentation:
            if gripper_segment_bounds is None:
                gripper_segment_bounds = self._compute_gripper_segment_bounds(
                    train_data, episode_ends, min_segment_length
                )
            self.gripper_segment_bounds = gripper_segment_bounds
        else:
            self.gripper_segment_bounds = None
        
        # Compute segment bounds (legacy, for backward compatibility)
        if segment_bounds is None and use_contact_segmentation:
            segment_bounds = self._compute_segment_bounds(
                train_data, episode_ends, contact_threshold, min_segment_length
            )
        elif segment_bounds is None:
            # Fall back to treating each episode as a single segment
            segment_bounds = {}
            current_start = 0
            for i, end_idx in enumerate(episode_ends):
                demo_id = f"demo_{i}"
                segment_bounds[demo_id] = [(0, end_idx - current_start)]
                current_start = end_idx
        
        # Use simple sample indices creation (same as PushTStateDataset)
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )
        
        # Compute statistics and normalize data
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])
        
        # Build mapping from sample index to reference frame for each gripper
        if use_gripper_segmentation:
            self.reference_frame_maps = {}  # Dict mapping gripper_id to reference_frame_map
            for gripper_id in range(self.num_pickers):
                self.reference_frame_maps[gripper_id] = self._build_gripper_reference_frame_map(
                    indices, episode_ends, gripper_segment_bounds, gripper_id
                )
        else:
            # Legacy: single reference frame map
            self.reference_frame_map = self._build_reference_frame_map(
                indices, episode_ends, segment_bounds
            )
            self.reference_frame_maps = None
        
        # Store attributes
        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.segment_bounds = segment_bounds
        self.episode_ends = episode_ends
    
    def _compute_segment_bounds(
        self,
        train_data: Dict[str, np.ndarray],
        episode_ends: np.ndarray,
        contact_threshold: float = 0.1,
        min_segment_length: int = 5,
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Compute segment bounds using environment state movement.
        
        Args:
            train_data: Dictionary with 'action' and 'obs' arrays.
            episode_ends: Array of episode end indices.
            contact_threshold: Threshold for detecting object movement.
            min_segment_length: Minimum length for a valid segment.
        
        Returns:
            Dictionary mapping demo_id to list of (start, end) segment bounds.
        """
        segment_bounds = {}
        current_start = 0
        
        for i, end_idx in enumerate(episode_ends):
            demo_id = f"demo_{i}"
            
            # Extract states and actions for this episode
            episode_states = train_data['obs'][current_start:end_idx]
            episode_actions = train_data['action'][current_start:end_idx]
            
            # Perform movement-based segmentation
            segments = segment_trajectory_by_env_movement(
                states=episode_states,
                actions=episode_actions,
                env_state_start_idx=self.env_state_start_idx,
                contact_threshold=contact_threshold,
                min_segment_length=min_segment_length,
            )
            
            segment_bounds[demo_id] = segments
            current_start = end_idx
        
        return segment_bounds
    
    def _compute_gripper_segment_bounds(
        self,
        train_data: Dict[str, np.ndarray],
        episode_ends: np.ndarray,
        min_segment_length: int = 5,
    ) -> Dict[str, Dict[int, List[Tuple[int, int]]]]:
        """Compute segment bounds using gripper grasp/release events.
        
        Args:
            train_data: Dictionary with 'action' and 'obs' arrays.
            episode_ends: Array of episode end indices.
            min_segment_length: Minimum length for a valid segment.
        
        Returns:
            Dictionary mapping demo_id to dict mapping gripper_id to list of (start, end) segment bounds.
        """
        gripper_segment_bounds = {}
        current_start = 0
        
        for i, end_idx in enumerate(episode_ends):
            demo_id = f"demo_{i}"
            
            # Extract actions for this episode
            episode_actions = train_data['action'][current_start:end_idx]
            
            # Perform gripper-based segmentation
            gripper_segments = segment_trajectory_by_gripper_events(
                actions=episode_actions,
                num_pickers=self.num_pickers,
                min_segment_length=min_segment_length,
            )
            
            gripper_segment_bounds[demo_id] = gripper_segments
            current_start = end_idx
        
        return gripper_segment_bounds
    
    def _build_gripper_reference_frame_map(
        self,
        indices: np.ndarray,
        episode_ends: np.ndarray,
        gripper_segment_bounds: Dict[str, Dict[int, List[Tuple[int, int]]]],
        gripper_id: int,
    ) -> np.ndarray:
        """Build mapping from sample index to reference frame index for a specific gripper.
        
        If the gripper's state doesn't change after the current segment (i.e., it's the last
        segment for that gripper), the reference is set to the end of the episode.
        
        Args:
            indices: Array of sample indices [buffer_start_idx, buffer_end_idx, ...]
            episode_ends: Array of episode end indices.
            gripper_segment_bounds: Dictionary mapping demo_id to dict mapping gripper_id to segments.
            gripper_id: Which gripper to build reference frames for.
        
        Returns:
            Array of reference frame indices, one per sample.
        """
        reference_frames = []
        
        for idx in range(len(indices)):
            buffer_start_idx = indices[idx, 0]
            
            # Find which episode this sample belongs to
            episode_idx = 0
            episode_start = 0
            episode_end = 0
            for i, end_idx in enumerate(episode_ends):
                if buffer_start_idx < end_idx:
                    episode_idx = i
                    episode_start = episode_ends[i - 1] if i > 0 else 0
                    episode_end = end_idx
                    break
            
            demo_id = f"demo_{episode_idx}"
            gripper_segments = gripper_segment_bounds.get(demo_id, {}).get(gripper_id, [])
            
            if len(gripper_segments) == 0:
                # No segments, use end of episode as reference
                ref_frame_idx = episode_end - 1
            else:
                # Find which segment this sample belongs to
                rel_start = buffer_start_idx - episode_start
                
                segment_idx = None
                for seg_idx, (seg_start, seg_end) in enumerate(gripper_segments):
                    if seg_start <= rel_start < seg_end:
                        segment_idx = seg_idx
                        break
                
                if segment_idx is None:
                    # Sample doesn't fall in any segment, use end of episode
                    ref_frame_idx = episode_end - 1
                else:
                    # Check if there's a next segment for this gripper
                    if segment_idx + 1 < len(gripper_segments):
                        # Next segment exists, use its start as reference
                        next_seg_start, _ = gripper_segments[segment_idx + 1]
                        ref_frame_idx = episode_start + next_seg_start
                    else:
                        # This is the last segment for this gripper
                        # Check if the gripper state changes after this segment
                        # by checking if the segment end is before the episode end
                        _, seg_end = gripper_segments[segment_idx]
                        if seg_end < episode_end - episode_start:
                            # There's more episode after this segment, but no state change
                            # Use the end of the episode as reference
                            ref_frame_idx = episode_end - 1
                        else:
                            # Segment goes to the end, use end of episode
                            ref_frame_idx = episode_end - 1
            
            reference_frames.append(ref_frame_idx)
        
        return np.array(reference_frames)
    
    def _build_reference_frame_map(
        self,
        indices: np.ndarray,
        episode_ends: np.ndarray,
        segment_bounds: Dict[str, List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Build mapping from sample index to reference frame index.
        
        For each sample, find which segment it belongs to based on its buffer_start_idx.
        The reference frame is the first frame of the next segment (or end of current
        segment if it's the last segment).
        
        Args:
            indices: Array of sample indices [buffer_start_idx, buffer_end_idx, ...]
            episode_ends: Array of episode end indices.
            segment_bounds: Dictionary mapping demo_id to list of (start, end) segment bounds.
        
        Returns:
            Array of reference frame indices, one per sample.
        """
        reference_frames = []
        
        for idx in range(len(indices)):
            buffer_start_idx = indices[idx, 0]
            
            # Find which episode this sample belongs to
            episode_idx = 0
            episode_start = 0
            for i, end_idx in enumerate(episode_ends):
                if buffer_start_idx < end_idx:
                    episode_idx = i
                    episode_start = episode_ends[i - 1] if i > 0 else 0
                    break
            
            demo_id = f"demo_{episode_idx}"
            segments = segment_bounds.get(demo_id, [])
            
            if len(segments) == 0:
                # No segments, use end of episode as reference
                ref_frame_idx = episode_ends[episode_idx] - 1
            else:
                # Find which segment this sample belongs to
                # Convert buffer_start_idx to relative position within episode
                rel_start = buffer_start_idx - episode_start
                
                # Find the segment that contains this position
                segment_idx = None
                for seg_idx, (seg_start, seg_end) in enumerate(segments):
                    if seg_start <= rel_start < seg_end:
                        segment_idx = seg_idx
                        break
                
                if segment_idx is None:
                    # Sample doesn't fall in any segment, use end of episode
                    ref_frame_idx = episode_ends[episode_idx] - 1
                else:
                    # Reference frame is the first frame of the next segment
                    # (or end of episode if it's the last segment)
                    if segment_idx + 1 < len(segments):
                        # Next segment exists, use its start as reference
                        next_seg_start, _ = segments[segment_idx + 1]
                        ref_frame_idx = episode_start + next_seg_start
                    else:
                        # Last segment, use end of episode as reference
                        ref_frame_idx = episode_ends[episode_idx] - 1
            
            reference_frames.append(ref_frame_idx)
        
        return np.array(reference_frames)
    
    def _filter_episode_stationary_final_states(
        self,
        episode_actions: np.ndarray,
        episode_obs: np.ndarray,
        action_threshold: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter out final frames from an episode where grippers have stopped moving.
        
        This prevents training on final states where actions are always zero, which can
        bias the model towards predicting zero actions.
        
        Args:
            episode_actions: Action array for a single episode, shape (T, act_dim).
            episode_obs: Observation array for a single episode, shape (T, obs_dim).
            action_threshold: Threshold for detecting movement (L2 norm of action).
        
        Returns:
            Tuple of (filtered_actions, filtered_obs) with final stationary frames removed.
        """
        if len(episode_actions) == 0:
            return episode_actions, episode_obs
        
        # Compute action magnitude for each frame (L2 norm of translation components)
        # Actions are [dx1, dy1, dz1, pick1, dx2, dy2, dz2, pick2, ...]
        # We only consider translation components (dx, dy, dz) for each gripper
        action_magnitudes = []
        for frame_idx in range(len(episode_actions)):
            frame_action = episode_actions[frame_idx]
            # Compute magnitude for each gripper's translation
            max_gripper_magnitude = 0.0
            for gripper_id in range(self.num_pickers):
                # Translation components are at indices: gripper_id * 4 + [0, 1, 2]
                dx_idx = gripper_id * 4
                dy_idx = gripper_id * 4 + 1
                dz_idx = gripper_id * 4 + 2
                translation = frame_action[[dx_idx, dy_idx, dz_idx]]
                magnitude = np.linalg.norm(translation)
                max_gripper_magnitude = max(max_gripper_magnitude, magnitude)
            action_magnitudes.append(max_gripper_magnitude)
        
        # Find the last frame where any gripper is still moving
        last_moving_frame = len(episode_actions) - 1
        for frame_idx in range(len(episode_actions) - 1, -1, -1):
            if action_magnitudes[frame_idx] > action_threshold:
                last_moving_frame = frame_idx
                break
        
        # Keep frames up to and including the last moving frame
        # Add 1 because slicing is exclusive of end index
        filtered_actions = episode_actions[:last_moving_frame + 1]
        filtered_obs = episode_obs[:last_moving_frame + 1]
        
        num_filtered = len(episode_actions) - len(filtered_actions)
        if num_filtered > 0:
            print(f"  Filtered out {num_filtered} final stationary frames from episode "
                  f"({100.0 * num_filtered / len(episode_actions):.1f}% of episode)")
        
        return filtered_actions, filtered_obs
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with reference frame(s).
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with 'obs', 'action', and 'reference_pos' keys.
            If use_gripper_segmentation=True, 'reference_pos' has shape (num_pickers, 3).
            Otherwise, 'reference_pos' has shape (2,).
        """
        # Get the indices for this datapoint (same as PushTStateDataset)
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        
        # Get normalized data (same as PushTStateDataset)
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        
        # Discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]
        
        if self.use_gripper_segmentation:
            # Extract reference positions for each gripper
            # Agent state contains picker positions: [x1, y1, z1, x2, y2, z2, ...]
            # We extract x, y, z for each gripper (all 3 dims per gripper)
            ref_positions = []
            for gripper_id in range(self.num_pickers):
                ref_frame_idx = self.reference_frame_maps[gripper_id][idx]
                ref_frame_idx = max(0, min(ref_frame_idx, len(self.normalized_train_data['obs']) - 1))
                
                # Get picker position for this gripper (x, y, z coordinates)
                picker_start_idx = gripper_id * 3  # Each picker has 3 dims (x, y, z)
                picker_pos_normalized = self.normalized_train_data['obs'][ref_frame_idx, picker_start_idx:picker_start_idx + 3]
                ref_positions.append(picker_pos_normalized)
            
            # Stack to shape (num_pickers, 3)
            nsample['reference_pos'] = np.stack(ref_positions, axis=0).astype(np.float32)
        else:
            # Legacy: single reference position (first 2 dims of agent state)
            ref_frame_idx = self.reference_frame_map[idx]
            ref_frame_idx = max(0, min(ref_frame_idx, len(self.normalized_train_data['obs']) - 1))
            ref_pos_normalized = self.normalized_train_data['obs'][ref_frame_idx, :2]
            nsample['reference_pos'] = ref_pos_normalized.astype(np.float32)
        
        return nsample


if __name__ == "__main__":
    # Example usage
    dataset_path = "datasets/ropeflatten_expert.zarr"
    
    # Parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    
    print("=" * 60)
    print("SoftGymSegmentedDatasetSimple")
    print("=" * 60)
    
    # Create dataset
    dataset = SoftGymSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_contact_segmentation=True,
        contact_threshold=0.1,
        min_segment_length=5,
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Number of segments per episode: {[len(segs) for segs in dataset.segment_bounds.values()]}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Visualize data in batch
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print(f"  obs: {batch['obs'].shape}")
    print(f"  action: {batch['action'].shape}")
    print(f"  reference_pos: {batch['reference_pos'].shape}")
