"""PushT state dataset with segmentation support for local flow policy training.

This dataset extends PushTStateDataset to:
1. Segment trajectories using velocity-based segmentation
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

# Add path for segmentation imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPATIALDECOMP_PATH = os.path.join(REPO_ROOT, "spatialdecomposition")
if SPATIALDECOMP_PATH not in sys.path:
    sys.path.append(SPATIALDECOMP_PATH)

def segment_trajectory_by_contact(
    states: np.ndarray,
    actions: np.ndarray,
    contact_threshold: float = 0.1,
    min_segment_length: int = 5
) -> List[Tuple[int, int]]:
    """Segment a trajectory based on contact with the object.
    
    Args:
        states: State array, shape (T, obs_dim). Assumes states contain:
            - states[:, :2]: agent position
            - states[:, 2:4]: block position (block_x, block_y)
            - states[:, 4:5]: block angle
        actions: Action array, shape (T, act_dim)
        contact_threshold: Threshold for detecting block movement (contact)
        min_segment_length: Minimum length for a valid segment
    
    Returns:
        List of (start, end) tuples for each segment.
    """
    T = len(states)
    segments = []
    
    # Calculate block movement (position and angle changes)
    # For PushT, block position is typically at indices 2:4
    if states.shape[1] >= 5:
        block_pos = states[:, 2:4]  # block_x, block_y
        block_angle = states[:, 4:5]  # block_angle
    else:
        # Fallback: use agent position if block info not available
        block_pos = states[:, :2]
        block_angle = np.zeros((T, 1))
    
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
    
    current_start = 0
    current_contact = in_contact[0]
    
    for i in range(1, T):
        if in_contact[i] != current_contact:
            # Contact state changed, end current segment
            if i - current_start >= min_segment_length:
                segment_starts.append(current_start)
                segment_ends.append(i)
            
            # Start new segment
            current_start = i
            current_contact = in_contact[i]
    
    # Add final segment
    if T - current_start >= min_segment_length:
        segment_starts.append(current_start)
        segment_ends.append(T)
    
    # Create segment bounds list
    for start, end in zip(segment_starts, segment_ends):
        segments.append((start, end))
    
    return segments


def create_segmented_sample_indices(
    episode_ends: np.ndarray,
    segment_bounds: Dict[str, List[Tuple[int, int]]],
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
    cross_segment_padding: int = 0,
    min_segment_length: int = 1,
    action_horizon: int = 1,
) -> np.ndarray:
    """Create sample indices for segmented trajectories with cross-segment padding.
    
    Args:
        episode_ends: Array of episode end indices (one-past-last).
        segment_bounds: Dictionary mapping demo_id to list of (start, end) segment bounds.
        sequence_length: Desired sequence length for each sample.
        pad_before: Number of padding steps before sequence.
        pad_after: Number of padding steps after sequence.
        cross_segment_padding: Number of steps to pad from next segment if available.
        
    Returns:
        Array of indices [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, segment_idx, reference_pos_idx].
    """
    indices = list()
    demo_id_to_start_idx = {}
    
    # Build mapping from demo_id to start index in concatenated buffer
    current_start = 0
    for i, end_idx in enumerate(episode_ends):
        demo_id = f"demo_{i}"
        demo_id_to_start_idx[demo_id] = current_start
        current_start = end_idx
    
    # Iterate over each episode
    for i in range(len(episode_ends)):
        demo_id = f"demo_{i}"
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        # Get segments for this episode
        if demo_id not in segment_bounds:
            # No segments available, fall back to treating whole episode as one segment
            segment_bounds[demo_id] = [(0, episode_length)]
        
        segments = segment_bounds[demo_id]
        
        # For each segment
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            # Skip segments that are too short
            if seg_end - seg_start < min_segment_length:
                continue
            
            # Segment boundaries in buffer coordinates (absolute indices)
            seg_start_buffer = start_idx + seg_start
            seg_end_buffer = start_idx + seg_end
            
            # Reference position is at the division point (end of segment)
            # This is the same for all samples in this segment
            ref_pos_idx = start_idx + seg_end - 1  # Reference is at division point (end of segment)
            
            # Check if next segment exists for cross-segment padding
            # We need to pad with (pred_horizon - action_horizon) steps from the next segment
            # This allows samples to extend beyond the reference frame by that amount
            next_seg_padding = 0
            if seg_idx + 1 < len(segments):
                next_seg_start, next_seg_end = segments[seg_idx + 1]
                # Copy the next (pred_horizon - action_horizon) steps from the next segment
                # This is the cross_segment_padding amount
                available_in_next_seg = next_seg_end - next_seg_start
                next_seg_padding = min(cross_segment_padding, available_in_next_seg)
            
            # Effective segment end including cross-segment padding
            # This allows samples to extend (pred_horizon - action_horizon) steps into the next segment
            effective_seg_end = min(seg_end_buffer + next_seg_padding, end_idx)
            
            # Create samples within this segment
            # Samples can start from the beginning of the segment (with pad_before for obs_horizon)
            # and can extend up to effective_seg_end (which includes padding into next segment)
            min_start = seg_start_buffer - pad_before
            max_start = effective_seg_end - sequence_length + pad_after
            
            # Ensure we have valid range
            if max_start < min_start:
                continue
            
            # Generate indices for this segment
            # idx is relative to episode start (can be negative for padding)
            for idx in range(min_start, max_start + 1):
                # Calculate buffer indices following the same pattern as create_sample_indices
                # buffer_start_idx: actual start in full buffer (absolute)
                buffer_start_idx = max(idx, seg_start_buffer - start_idx) + start_idx
                # buffer_end_idx: actual end in full buffer (absolute)
                buffer_end_idx = min(idx + sequence_length, effective_seg_end - start_idx) + start_idx
                
                # Ensure buffer indices are within episode bounds
                buffer_start_idx = max(buffer_start_idx, start_idx)
                buffer_end_idx = min(buffer_end_idx, end_idx)
                
                # Ensure we have a valid buffer range
                if buffer_end_idx <= buffer_start_idx:
                    continue
                
                # Calculate offsets following the exact same logic as create_sample_indices
                start_offset = buffer_start_idx - (idx + start_idx)
                end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
                
                # Calculate sample indices (where in the output sequence to place the sample)
                sample_start_idx = 0 + start_offset
                sample_end_idx = sequence_length - end_offset
                
                # Ensure valid sample range
                if sample_end_idx <= sample_start_idx:
                    continue
                
                # Verify the sample we extract will fit in the target slice
                actual_sample_length = buffer_end_idx - buffer_start_idx
                expected_sample_length = sample_end_idx - sample_start_idx
                
                # They must match for the assignment to work
                if actual_sample_length != expected_sample_length:
                    # This shouldn't happen with correct logic, but if it does, skip
                    continue
                
                # All samples in this segment use the same reference position
                # The reference position is at the division point (end of segment)
                # This is fixed for all samples in the segment
                indices.append([
                    buffer_start_idx,
                    buffer_end_idx,
                    sample_start_idx,
                    sample_end_idx,
                    seg_idx,  # Segment index within episode
                    ref_pos_idx,  # Reference position index (same for all samples in segment)
                ])
    
    indices = np.array(indices)
    return indices


class PushTSegmentedDataset(PushTStateDataset):
    """Dataset for PushT with segmentation support and reference position extraction."""
    
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        max_demos: Optional[int] = None,
        demo_indices: Optional[List[int]] = None,
        use_contact_segmentation: bool = True,
        contact_threshold: float = 0.1,
        min_segment_length: int = 5,
        segment_bounds: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ):
        """Initialize PushT segmented dataset.
        
        Args:
            dataset_path: Path to zarr dataset file.
            pred_horizon: Number of steps to predict ahead.
            obs_horizon: Number of observation steps to use.
            action_horizon: Number of action steps to execute.
            max_demos: Maximum number of episodes to use (None = all). Ignored if demo_indices is provided.
            demo_indices: Optional list of episode indices to use (e.g., [0, 1, 5, 10]). 
                         If provided, only these episodes will be used.
            use_contact_segmentation: Whether to use contact-based segmentation.
            contact_threshold: Threshold for detecting block movement (contact).
            min_segment_length: Minimum length for a valid segment.
            segment_bounds: Optional pre-computed segment bounds. If None, will compute from contact.
        
        Note:
            cross_segment_padding is automatically computed as pred_horizon - action_horizon
            to handle predictions that extend beyond the current segment.
        """
        # Compute cross-segment padding automatically
        # This is the number of steps beyond action_horizon that we predict
        cross_segment_padding = pred_horizon - action_horizon
        # First, load the basic dataset structure
        dataset_root = zarr.open(dataset_path, 'r')
        all_train_data = {
            'action': dataset_root['data']['action'][:],
            'obs': dataset_root['data']['state'][:]
        }
        all_episode_ends = dataset_root['meta']['episode_ends'][:]
        
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
            # Truncate data accordingly
            if len(episode_ends) > 0:
                max_idx = episode_ends[-1]
                train_data = {
                    'action': all_train_data['action'][:max_idx],
                    'obs': all_train_data['obs'][:max_idx]
                }
            else:
                train_data = all_train_data
            # Store demo indices used (0 to max_demos-1)
            self.demo_indices = list(range(max_demos))
        else:
            # Use all episodes
            episode_ends = all_episode_ends
            train_data = all_train_data
            # Store demo indices used (all)
            self.demo_indices = list(range(len(all_episode_ends)))
        
        # Store segmentation parameters
        self.contact_threshold = contact_threshold
        self.min_segment_length = min_segment_length
        
        # Compute segment bounds
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
        
        # Create sample indices with segmentation
        # Note: pad_after should allow samples to extend (pred_horizon - action_horizon) steps beyond reference
        # The reference is at: sample_end + action_horizon
        # So samples should be able to extend up to: reference + (pred_horizon - action_horizon)
        # Which means samples can end up to: reference - action_horizon + (pred_horizon - action_horizon) = reference + pred_horizon - 2*action_horizon
        # Actually, we want samples to end such that sample_end + action_horizon = ref, and sample extends pred_horizon steps
        # So sample_end can be up to ref - action_horizon, and sample extends pred_horizon steps from there
        # The pad_after should allow extending beyond the segment boundary by cross_segment_padding
        indices = create_segmented_sample_indices(
            episode_ends=episode_ends,
            segment_bounds=segment_bounds,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=cross_segment_padding,  # Allow extending (pred_horizon - action_horizon) steps beyond reference
            cross_segment_padding=cross_segment_padding,
            min_segment_length=min_segment_length,
            action_horizon=action_horizon,  # Needed to calculate reference position correctly
        )
        
        # Compute statistics and normalize data
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])
        
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
        """Compute segment bounds using contact-based segmentation.
        
        Args:
            train_data: Dictionary with 'action' and 'obs' arrays.
            episode_ends: Array of episode end indices.
            contact_threshold: Threshold for detecting block movement (contact).
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
            
            # Perform contact-based segmentation
            segments = segment_trajectory_by_contact(
                states=episode_states,
                actions=episode_actions,
                contact_threshold=contact_threshold,
                min_segment_length=min_segment_length,
            )
            
            segment_bounds[demo_id] = segments
            current_start = end_idx
        
        return segment_bounds
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with reference position.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with 'obs', 'action', and 'reference_pos' keys.
        """
        # Get the indices for this datapoint
        (buffer_start_idx, buffer_end_idx,
         sample_start_idx, sample_end_idx,
         seg_idx, ref_pos_idx) = self.indices[idx]
        
        # Get normalized data
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
        
        # Extract reference position (last point of segment)
        # Reference position is in the observation space (state)
        # For PushT, state typically contains [agent_pos_x, agent_pos_y, ...]
        ref_pos = self.normalized_train_data['obs'][ref_pos_idx, :2]  # First 2 dims are position
        
        nsample['reference_pos'] = ref_pos.astype(np.float32)
        
        return nsample


class PushTSegmentedDatasetSimple(PushTStateDataset):
    """Simplified dataset for PushT with segmentation support.
    
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
        contact_threshold: float = 0.1,
        min_segment_length: int = 5,
        segment_bounds: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ):
        """Initialize simplified PushT segmented dataset.
        
        Args:
            dataset_path: Path to zarr dataset file.
            pred_horizon: Number of steps to predict ahead.
            obs_horizon: Number of observation steps to use.
            action_horizon: Number of action steps to execute.
            max_demos: Maximum number of episodes to use (None = all). Ignored if demo_indices is provided.
            demo_indices: Optional list of episode indices to use (e.g., [0, 1, 5, 10]). 
                         If provided, only these episodes will be used.
            use_contact_segmentation: Whether to use contact-based segmentation.
            contact_threshold: Threshold for detecting block movement (contact).
            min_segment_length: Minimum length for a valid segment.
            segment_bounds: Optional pre-computed segment bounds. If None, will compute from contact.
        """
        # First, load the basic dataset structure
        dataset_root = zarr.open(dataset_path, 'r')
        all_train_data = {
            'action': dataset_root['data']['action'][:],
            'obs': dataset_root['data']['state'][:]
        }
        all_episode_ends = dataset_root['meta']['episode_ends'][:]
        
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
            # Truncate data accordingly
            if len(episode_ends) > 0:
                max_idx = episode_ends[-1]
                train_data = {
                    'action': all_train_data['action'][:max_idx],
                    'obs': all_train_data['obs'][:max_idx]
                }
            else:
                train_data = all_train_data
            # Store demo indices used (0 to max_demos-1)
            self.demo_indices = list(range(max_demos))
        else:
            # Use all episodes
            episode_ends = all_episode_ends
            train_data = all_train_data
            # Store demo indices used (all)
            self.demo_indices = list(range(len(all_episode_ends)))
        
        # Store segmentation parameters
        self.contact_threshold = contact_threshold
        self.min_segment_length = min_segment_length
        
        # Compute segment bounds
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
        
        # Build mapping from sample index to reference frame
        # For each sample, find which segment it belongs to and assign the next frame
        self.reference_frame_map = self._build_reference_frame_map(
            indices, episode_ends, segment_bounds
        )
        
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
        """Compute segment bounds using contact-based segmentation.
        
        Args:
            train_data: Dictionary with 'action' and 'obs' arrays.
            episode_ends: Array of episode end indices.
            contact_threshold: Threshold for detecting block movement (contact).
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
            
            # Perform contact-based segmentation
            segments = segment_trajectory_by_contact(
                states=episode_states,
                actions=episode_actions,
                contact_threshold=contact_threshold,
                min_segment_length=min_segment_length,
            )
            
            segment_bounds[demo_id] = segments
            current_start = end_idx
        
        return segment_bounds
    
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
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with reference frame.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary with 'obs', 'action', and 'reference_pos' keys.
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
        
        # Extract reference position from the assigned frame
        ref_frame_idx = self.reference_frame_map[idx]
        # Clamp to valid range
        ref_frame_idx = max(0, min(ref_frame_idx, len(self.normalized_train_data['obs']) - 1))
        ref_pos = self.normalized_train_data['obs'][ref_frame_idx, :2]  # First 2 dims are position
        
        nsample['reference_pos'] = ref_pos.astype(np.float32)
        
        return nsample


if __name__ == "__main__":
    # Example usage
    dataset_path = "datasets/pusht_cchi_v7_replay.zarr.zip"
    
    # Parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    
    print("=" * 60)
    print("Original PushTSegmentedDataset (complex)")
    print("=" * 60)
    
    # Create original segmented dataset (complex version)
    dataset_complex = PushTSegmentedDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_contact_segmentation=True,
        contact_threshold=0.1,
        min_segment_length=5,
    )
    
    print(f"Dataset created with {len(dataset_complex)} samples")
    print(f"Number of segments per episode: {[len(segs) for segs in dataset_complex.segment_bounds.values()]}")
    
    # Create dataloader
    dataloader_complex = torch.utils.data.DataLoader(
        dataset_complex,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Visualize data in batch
    batch_complex = next(iter(dataloader_complex))
    print("\nBatch shapes:")
    print(f"  obs: {batch_complex['obs'].shape}")
    print(f"  action: {batch_complex['action'].shape}")
    print(f"  reference_pos: {batch_complex['reference_pos'].shape}")
    
    print("\n" + "=" * 60)
    print("New PushTSegmentedDatasetSimple (simplified)")
    print("=" * 60)
    
    # Create simplified segmented dataset
    dataset_simple = PushTSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        use_contact_segmentation=True,
        contact_threshold=0.1,
        min_segment_length=5,
    )
    
    print(f"Dataset created with {len(dataset_simple)} samples")
    print(f"Number of segments per episode: {[len(segs) for segs in dataset_simple.segment_bounds.values()]}")
    
    # Create dataloader
    dataloader_simple = torch.utils.data.DataLoader(
        dataset_simple,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Visualize data in batch
    batch_simple = next(iter(dataloader_simple))
    print("\nBatch shapes:")
    print(f"  obs: {batch_simple['obs'].shape}")
    print(f"  action: {batch_simple['action'].shape}")
    print(f"  reference_pos: {batch_simple['reference_pos'].shape}")
    
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"Complex dataset samples: {len(dataset_complex)}")
    print(f"Simple dataset samples: {len(dataset_simple)}")
    print(f"Sample count difference: {len(dataset_complex) - len(dataset_simple)}")

