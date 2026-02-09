#!/usr/bin/env python3
"""Analysis script to check the randomization of initial states in the PushT dataset.

This script analyzes the distribution of initial states across episodes to verify
that the dataset has good randomization of starting conditions.

Usage:
    python SD_pusht/scripts/analyze_initial_state_randomization.py \
        --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
        --max-demos 200 \
        --output-dir analysis_results
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist
import zarr

# Try to import gym_pusht for rendering (optional)
try:
    import gymnasium as gym
    import gym_pusht
    GYM_PUSHT_AVAILABLE = True
except ImportError:
    GYM_PUSHT_AVAILABLE = False
    print("Warning: gym_pusht not available. Initial state visualization will be skipped.")

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze initial state randomization in PushT dataset"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/pusht_cchi_v7_replay.zarr.zip",
        help="Path to zarr dataset file"
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Maximum number of episodes to analyze (None = all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results and plots"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-visualization-samples",
        type=int,
        default=16,
        help="Number of initial states to visualize as images (default: 16)"
    )
    return parser.parse_args()


def load_initial_states(dataset_path, max_demos=None):
    """Load initial states from the dataset.
    
    Args:
        dataset_path: Path to zarr dataset file
        max_demos: Maximum number of episodes to load (None = all)
    
    Returns:
        initial_states: Array of shape (n_episodes, 5) containing initial states
            [agent_x, agent_y, block_x, block_y, block_angle]
        episode_ends: Array of episode end indices
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset_root = zarr.open(dataset_path, 'r')
    
    # Load observations (states)
    obs = dataset_root['data']['state'][:]
    episode_ends = dataset_root['meta']['episode_ends'][:]
    
    if max_demos is not None:
        episode_ends = episode_ends[:max_demos]
        if len(episode_ends) > 0:
            max_idx = episode_ends[-1]
            obs = obs[:max_idx]
    
    # Extract initial state for each episode
    initial_states = []
    start_idx = 0
    
    for end_idx in episode_ends:
        # First observation of the episode
        initial_state = obs[start_idx]
        initial_states.append(initial_state)
        start_idx = end_idx
    
    initial_states = np.array(initial_states)
    
    print(f"Loaded {len(initial_states)} episodes")
    print(f"Initial states shape: {initial_states.shape}")
    
    return initial_states, episode_ends


def load_goal_positions(dataset_path, initial_states, max_demos=None):
    """Load goal positions from the environment for each episode.
    
    Args:
        dataset_path: Path to zarr dataset file (for reference)
        initial_states: Array of initial states for each episode
        max_demos: Maximum number of episodes to load (None = all)
    
    Returns:
        goal_positions: Array of shape (n_episodes, 3) containing goal poses
            [goal_x, goal_y, goal_angle]
    """
    if not GYM_PUSHT_AVAILABLE:
        print("Warning: gym_pusht not available. Cannot extract goal positions.")
        return None
    
    print(f"\nExtracting goal positions from environment...")
    
    try:
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    except Exception as e:
        print(f"Warning: Could not create PushT environment: {e}")
        return None
    
    goal_positions = []
    n_episodes = len(initial_states)
    
    for i in range(n_episodes):
        if max_demos is not None and i >= max_demos:
            break
            
        try:
            # Reset environment to initial state to get goal position
            initial_state = initial_states[i]
            env.reset(options={"reset_to_state": initial_state.tolist()})
            
            # Get goal pose from info
            info = env._get_info()
            goal_pose = info.get("goal_pose", None)
            
            if goal_pose is not None:
                goal_positions.append(goal_pose)
            else:
                # Fallback: use default goal pose if not in info
                goal_positions.append(np.array([256, 256, np.pi / 4]))
                
        except Exception as e:
            print(f"Warning: Could not get goal for episode {i}: {e}")
            # Use default goal pose as fallback
            goal_positions.append(np.array([256, 256, np.pi / 4]))
    
    env.close()
    
    goal_positions = np.array(goal_positions)
    print(f"Extracted {len(goal_positions)} goal positions")
    print(f"Goal positions shape: {goal_positions.shape}")
    
    return goal_positions


def compute_statistics(initial_states):
    """Compute statistical measures of initial state distribution.
    
    Args:
        initial_states: Array of shape (n_episodes, 5)
    
    Returns:
        Dictionary with statistics for each dimension
    """
    n_episodes = len(initial_states)
    dim_names = ['agent_x', 'agent_y', 'block_x', 'block_y', 'block_angle']
    
    stats_dict = {}
    
    for i, dim_name in enumerate(dim_names):
        values = initial_states[:, i]
        
        stats_dict[dim_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'range': np.max(values) - np.min(values),
            'coverage': (np.max(values) - np.min(values)) / 512.0 if i < 4 else (np.max(values) - np.min(values)) / (2 * np.pi),
        }
        
        # Test for uniform distribution (Kolmogorov-Smirnov test)
        if i < 4:  # Position dimensions (should be uniform in [0, 512])
            expected_min, expected_max = 0.0, 512.0
        else:  # Angle dimension (should be uniform in [0, 2*pi])
            expected_min, expected_max = 0.0, 2 * np.pi
        
        # Normalize to [0, 1] for KS test
        normalized_values = (values - expected_min) / (expected_max - expected_min)
        ks_statistic, p_value = stats.kstest(normalized_values, 'uniform')
        
        stats_dict[dim_name]['ks_statistic'] = ks_statistic
        stats_dict[dim_name]['ks_p_value'] = p_value
        stats_dict[dim_name]['is_uniform'] = p_value > 0.05  # Not significantly different from uniform
    
    return stats_dict


def compute_spatial_distribution(initial_states):
    """Compute spatial distribution metrics.
    
    Args:
        initial_states: Array of shape (n_episodes, 5)
    
    Returns:
        Dictionary with spatial distribution metrics
    """
    agent_positions = initial_states[:, :2]  # (n_episodes, 2)
    block_positions = initial_states[:, 2:4]  # (n_episodes, 2)
    
    # Compute pairwise distances
    agent_distances = pdist(agent_positions)
    block_distances = pdist(block_positions)
    
    # Compute coverage (area covered by points)
    agent_x_range = np.max(agent_positions[:, 0]) - np.min(agent_positions[:, 0])
    agent_y_range = np.max(agent_positions[:, 1]) - np.min(agent_positions[:, 1])
    agent_coverage_area = agent_x_range * agent_y_range
    agent_coverage_ratio = agent_coverage_area / (512.0 * 512.0)
    
    block_x_range = np.max(block_positions[:, 0]) - np.min(block_positions[:, 0])
    block_y_range = np.max(block_positions[:, 1]) - np.min(block_positions[:, 1])
    block_coverage_area = block_x_range * block_y_range
    block_coverage_ratio = block_coverage_area / (512.0 * 512.0)
    
    # Compute clustering metrics (mean nearest neighbor distance)
    # For each point, find distance to nearest neighbor
    agent_nn_distances = []
    block_nn_distances = []
    
    for i in range(len(agent_positions)):
        # Distance to all other points
        agent_dists = np.linalg.norm(agent_positions - agent_positions[i], axis=1)
        agent_dists[agent_dists == 0] = np.inf  # Exclude self
        agent_nn_distances.append(np.min(agent_dists))
        
        block_dists = np.linalg.norm(block_positions - block_positions[i], axis=1)
        block_dists[block_dists == 0] = np.inf
        block_nn_distances.append(np.min(block_dists))
    
    agent_mean_nn_distance = np.mean(agent_nn_distances)
    block_mean_nn_distance = np.mean(block_nn_distances)
    
    return {
        'agent': {
            'mean_pairwise_distance': np.mean(agent_distances),
            'std_pairwise_distance': np.std(agent_distances),
            'coverage_area': agent_coverage_area,
            'coverage_ratio': agent_coverage_ratio,
            'mean_nn_distance': agent_mean_nn_distance,
            'x_range': agent_x_range,
            'y_range': agent_y_range,
        },
        'block': {
            'mean_pairwise_distance': np.mean(block_distances),
            'std_pairwise_distance': np.std(block_distances),
            'coverage_area': block_coverage_area,
            'coverage_ratio': block_coverage_ratio,
            'mean_nn_distance': block_mean_nn_distance,
            'x_range': block_x_range,
            'y_range': block_y_range,
        }
    }


def compute_goal_statistics(goal_positions):
    """Compute statistical measures of goal position distribution.
    
    Args:
        goal_positions: Array of shape (n_episodes, 3) [goal_x, goal_y, goal_angle]
    
    Returns:
        Dictionary with statistics for each dimension
    """
    if goal_positions is None:
        return None
        
    n_episodes = len(goal_positions)
    dim_names = ['goal_x', 'goal_y', 'goal_angle']
    
    stats_dict = {}
    
    for i, dim_name in enumerate(dim_names):
        values = goal_positions[:, i]
        
        stats_dict[dim_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'range': np.max(values) - np.min(values),
            'coverage': (np.max(values) - np.min(values)) / 512.0 if i < 2 else (np.max(values) - np.min(values)) / (2 * np.pi),
        }
        
        # Test for uniform distribution (Kolmogorov-Smirnov test)
        if i < 2:  # Position dimensions (should be uniform in [0, 512])
            expected_min, expected_max = 0.0, 512.0
        else:  # Angle dimension (should be uniform in [0, 2*pi])
            expected_min, expected_max = 0.0, 2 * np.pi
        
        # Normalize to [0, 1] for KS test
        normalized_values = (values - expected_min) / (expected_max - expected_min)
        ks_statistic, p_value = stats.kstest(normalized_values, 'uniform')
        
        stats_dict[dim_name]['ks_statistic'] = ks_statistic
        stats_dict[dim_name]['ks_p_value'] = p_value
        stats_dict[dim_name]['is_uniform'] = p_value > 0.05  # Not significantly different from uniform
    
    return stats_dict


def compute_goal_spatial_distribution(goal_positions):
    """Compute spatial distribution metrics for goal positions.
    
    Args:
        goal_positions: Array of shape (n_episodes, 3) [goal_x, goal_y, goal_angle]
    
    Returns:
        Dictionary with spatial distribution metrics
    """
    if goal_positions is None:
        return None
        
    goal_pos = goal_positions[:, :2]  # (n_episodes, 2) - only x, y positions
    
    # Compute pairwise distances
    goal_distances = pdist(goal_pos)
    
    # Compute coverage (area covered by points)
    goal_x_range = np.max(goal_pos[:, 0]) - np.min(goal_pos[:, 0])
    goal_y_range = np.max(goal_pos[:, 1]) - np.min(goal_pos[:, 1])
    goal_coverage_area = goal_x_range * goal_y_range
    goal_coverage_ratio = goal_coverage_area / (512.0 * 512.0)
    
    # Compute clustering metrics (mean nearest neighbor distance)
    goal_nn_distances = []
    for i in range(len(goal_pos)):
        goal_dists = np.linalg.norm(goal_pos - goal_pos[i], axis=1)
        goal_dists[goal_dists == 0] = np.inf  # Exclude self
        goal_nn_distances.append(np.min(goal_dists))
    
    goal_mean_nn_distance = np.mean(goal_nn_distances)
    
    return {
        'mean_pairwise_distance': np.mean(goal_distances),
        'std_pairwise_distance': np.std(goal_distances),
        'coverage_area': goal_coverage_area,
        'coverage_ratio': goal_coverage_ratio,
        'mean_nn_distance': goal_mean_nn_distance,
        'x_range': goal_x_range,
        'y_range': goal_y_range,
    }


def visualize_distributions(initial_states, output_dir):
    """Create visualization plots for initial state distributions.
    
    Args:
        initial_states: Array of shape (n_episodes, 5)
        output_dir: Output directory for saving plots
    """
    dim_names = ['agent_x', 'agent_y', 'block_x', 'block_y', 'block_angle']
    
    # 1. Histograms for each dimension
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (dim_name, ax) in enumerate(zip(dim_names, axes)):
        values = initial_states[:, i]
        
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(dim_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {dim_name}')
        ax.axvline(np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(np.median(values), color='g', linestyle='--', label=f'Median: {np.median(values):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove the last empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_state_histograms.png'), dpi=150, bbox_inches='tight')
    print(f"Saved histogram plot to: {os.path.join(output_dir, 'initial_state_histograms.png')}")
    plt.close()
    
    # 2. Spatial scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Agent positions
    ax = axes[0]
    agent_positions = initial_states[:, :2]
    scatter = ax.scatter(agent_positions[:, 0], agent_positions[:, 1], 
                        alpha=0.6, s=20, c=range(len(agent_positions)), cmap='viridis')
    ax.set_xlabel('Agent X')
    ax.set_ylabel('Agent Y')
    ax.set_title(f'Agent Initial Positions (n={len(agent_positions)})')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Episode Index')
    
    # Block positions
    ax = axes[1]
    block_positions = initial_states[:, 2:4]
    scatter = ax.scatter(block_positions[:, 0], block_positions[:, 1], 
                        alpha=0.6, s=20, c=range(len(block_positions)), cmap='viridis')
    ax.set_xlabel('Block X')
    ax.set_ylabel('Block Y')
    ax.set_title(f'Block Initial Positions (n={len(block_positions)})')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Episode Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_state_spatial.png'), dpi=150, bbox_inches='tight')
    print(f"Saved spatial plot to: {os.path.join(output_dir, 'initial_state_spatial.png')}")
    plt.close()
    
    # 3. 2D density plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Agent positions density
    ax = axes[0]
    sns.kdeplot(x=agent_positions[:, 0], y=agent_positions[:, 1], 
                ax=ax, fill=True, cmap='Blues', alpha=0.6)
    ax.scatter(agent_positions[:, 0], agent_positions[:, 1], 
              alpha=0.3, s=10, c='red', edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Agent X')
    ax.set_ylabel('Agent Y')
    ax.set_title('Agent Initial Positions Density')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    
    # Block positions density
    ax = axes[1]
    sns.kdeplot(x=block_positions[:, 0], y=block_positions[:, 1], 
                ax=ax, fill=True, cmap='Reds', alpha=0.6)
    ax.scatter(block_positions[:, 0], block_positions[:, 1], 
              alpha=0.3, s=10, c='blue', edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Block X')
    ax.set_ylabel('Block Y')
    ax.set_title('Block Initial Positions Density')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_state_density.png'), dpi=150, bbox_inches='tight')
    print(f"Saved density plot to: {os.path.join(output_dir, 'initial_state_density.png')}")
    plt.close()
    
    # 4. Box plots for each dimension
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = [initial_states[:, i] for i in range(5)]
    bp = ax.boxplot(data_to_plot, labels=dim_names, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Value')
    ax.set_title('Initial State Distributions (Box Plots)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_state_boxplots.png'), dpi=150, bbox_inches='tight')
    print(f"Saved box plot to: {os.path.join(output_dir, 'initial_state_boxplots.png')}")
    plt.close()


def visualize_initial_state_images(initial_states, output_dir, n_samples=16):
    """Render and visualize a grid of initial states from the dataset.
    
    Args:
        initial_states: Array of shape (n_episodes, 5) containing initial states
        output_dir: Output directory for saving images
        n_samples: Number of initial states to visualize (will be arranged in a grid)
    """
    if not GYM_PUSHT_AVAILABLE:
        print("Skipping initial state image visualization (gym_pusht not available)")
        return
    
    print(f"\nRendering {n_samples} initial states...")
    
    # Sample diverse initial states
    n_episodes = len(initial_states)
    if n_samples > n_episodes:
        n_samples = n_episodes
        print(f"Warning: Requested {n_samples} samples but only {n_episodes} episodes available")
    
    # Select diverse samples - evenly spaced indices
    sample_indices = np.linspace(0, n_episodes - 1, n_samples, dtype=int)
    
    # Create environment
    try:
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    except Exception as e:
        print(f"Warning: Could not create PushT environment: {e}")
        print("Skipping initial state image visualization")
        return
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    rendered_count = 0
    for idx, sample_idx in enumerate(sample_indices):
        if idx >= len(axes):
            break
            
        initial_state = initial_states[sample_idx]
        
        try:
            # Reset environment to this initial state
            # Note: The state format is [agent_x, agent_y, block_x, block_y, block_angle]
            env.reset(options={"reset_to_state": initial_state.tolist()})
            
            # Render the state
            img = env.render()
            
            # Display the image
            ax = axes[idx]
            ax.imshow(img)
            ax.set_title(f"Episode {sample_idx}\n"
                        f"A:({initial_state[0]:.0f},{initial_state[1]:.0f}) "
                        f"B:({initial_state[2]:.0f},{initial_state[3]:.0f}) "
                        f"θ:{initial_state[4]:.2f}",
                        fontsize=8)
            ax.axis('off')
            rendered_count += 1
            
        except Exception as e:
            print(f"Warning: Could not render state {sample_idx}: {e}")
            axes[idx].axis('off')
            axes[idx].text(0.5, 0.5, f"Error\nEpisode {sample_idx}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
    
    # Hide unused subplots
    for idx in range(rendered_count, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Initial State Diversity Visualization (showing {rendered_count} of {n_samples} samples)',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'initial_state_images.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved initial state images to: {output_path}")
    plt.close()
    
    # Close environment
    env.close()
    
    # Also create a larger version with more samples if dataset is large enough
    if n_episodes >= 64:
        print("Creating additional visualization with 64 samples...")
        n_samples_large = 64
        sample_indices_large = np.linspace(0, n_episodes - 1, n_samples_large, dtype=int)
        n_cols_large = 8
        n_rows_large = 8
        
        fig, axes = plt.subplots(n_rows_large, n_cols_large, figsize=(n_cols_large * 2, n_rows_large * 2))
        axes = axes.flatten()
        
        try:
            env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
        except Exception as e:
            print(f"Warning: Could not create PushT environment for large visualization: {e}")
            return
        
        rendered_count_large = 0
        for idx, sample_idx in enumerate(sample_indices_large):
            if idx >= len(axes):
                break
                
            initial_state = initial_states[sample_idx]
            
            try:
                env.reset(options={"reset_to_state": initial_state.tolist()})
                img = env.render()
                ax = axes[idx]
                ax.imshow(img)
                ax.set_title(f"#{sample_idx}", fontsize=6)
                ax.axis('off')
                rendered_count_large += 1
            except Exception as e:
                axes[idx].axis('off')
        
        for idx in range(rendered_count_large, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Initial State Diversity - 64 Samples', fontsize=12, y=0.995)
        plt.tight_layout()
        
        output_path_large = os.path.join(output_dir, 'initial_state_images_64.png')
        plt.savefig(output_path_large, dpi=100, bbox_inches='tight')
        print(f"Saved large initial state images to: {output_path_large}")
        plt.close()
        
        env.close()


def visualize_goal_distributions(goal_positions, output_dir):
    """Create visualization plots for goal position distributions.
    
    Args:
        goal_positions: Array of shape (n_episodes, 3) [goal_x, goal_y, goal_angle]
        output_dir: Output directory for saving plots
    """
    if goal_positions is None:
        print("Skipping goal position visualizations (no goal data available)")
        return
    
    dim_names = ['goal_x', 'goal_y', 'goal_angle']
    
    # 1. Histograms for each dimension
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (dim_name, ax) in enumerate(zip(dim_names, axes)):
        values = goal_positions[:, i]
        
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(dim_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {dim_name}')
        ax.axvline(np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.2f}')
        ax.axvline(np.median(values), color='g', linestyle='--', label=f'Median: {np.median(values):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goal_position_histograms.png'), dpi=150, bbox_inches='tight')
    print(f"Saved goal histogram plot to: {os.path.join(output_dir, 'goal_position_histograms.png')}")
    plt.close()
    
    # 2. Spatial scatter plot for goal positions
    fig, ax = plt.subplots(figsize=(10, 10))
    
    goal_pos = goal_positions[:, :2]
    scatter = ax.scatter(goal_pos[:, 0], goal_pos[:, 1], 
                        alpha=0.6, s=30, c=range(len(goal_pos)), cmap='viridis')
    ax.set_xlabel('Goal X')
    ax.set_ylabel('Goal Y')
    ax.set_title(f'Goal Positions (n={len(goal_pos)})')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Episode Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goal_position_spatial.png'), dpi=150, bbox_inches='tight')
    print(f"Saved goal spatial plot to: {os.path.join(output_dir, 'goal_position_spatial.png')}")
    plt.close()
    
    # 3. 2D density plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sns.kdeplot(x=goal_pos[:, 0], y=goal_pos[:, 1], 
                ax=ax, fill=True, cmap='Purples', alpha=0.6)
    ax.scatter(goal_pos[:, 0], goal_pos[:, 1], 
              alpha=0.3, s=20, c='purple', edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Goal X')
    ax.set_ylabel('Goal Y')
    ax.set_title('Goal Position Density')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goal_position_density.png'), dpi=150, bbox_inches='tight')
    print(f"Saved goal density plot to: {os.path.join(output_dir, 'goal_position_density.png')}")
    plt.close()


def visualize_initial_vs_goal(initial_states, goal_positions, output_dir):
    """Create visualization comparing initial block positions and goal positions.
    
    Args:
        initial_states: Array of shape (n_episodes, 5)
        goal_positions: Array of shape (n_episodes, 3)
        output_dir: Output directory for saving plots
    """
    if goal_positions is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Extract initial block positions
    initial_block_pos = initial_states[:, 2:4]  # block_x, block_y
    goal_pos = goal_positions[:, :2]  # goal_x, goal_y
    
    # Plot initial block positions
    ax.scatter(initial_block_pos[:, 0], initial_block_pos[:, 1], 
              alpha=0.5, s=30, c='blue', label='Initial Block Positions', marker='o')
    
    # Plot goal positions
    ax.scatter(goal_pos[:, 0], goal_pos[:, 1], 
              alpha=0.5, s=30, c='purple', label='Goal Positions', marker='s')
    
    # Draw arrows from initial to goal (sample a subset to avoid clutter)
    n_samples = min(50, len(initial_block_pos))
    sample_indices = np.linspace(0, len(initial_block_pos) - 1, n_samples, dtype=int)
    
    for idx in sample_indices:
        ax.annotate('', xy=goal_pos[idx], xytext=initial_block_pos[idx],
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=1))
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Initial Block Positions vs Goal Positions\n(arrows show direction from initial to goal)')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_vs_goal_positions.png'), dpi=150, bbox_inches='tight')
    print(f"Saved initial vs goal plot to: {os.path.join(output_dir, 'initial_vs_goal_positions.png')}")
    plt.close()


def print_report(initial_states, stats_dict, spatial_stats, goal_stats_dict=None, goal_spatial_stats=None):
    """Print analysis report.
    
    Args:
        initial_states: Array of initial states
        stats_dict: Dictionary with statistics
        spatial_stats: Dictionary with spatial distribution metrics
        goal_stats_dict: Dictionary with goal position statistics (optional)
        goal_spatial_stats: Dictionary with goal spatial distribution metrics (optional)
    """
    print("\n" + "="*80)
    print("INITIAL STATE RANDOMIZATION ANALYSIS REPORT")
    print("="*80)
    
    n_episodes = len(initial_states)
    print(f"\nTotal number of episodes analyzed: {n_episodes}")
    
    print("\n" + "-"*80)
    print("DIMENSION-WISE STATISTICS")
    print("-"*80)
    
    dim_names = ['agent_x', 'agent_y', 'block_x', 'block_y', 'block_angle']
    for dim_name in dim_names:
        s = stats_dict[dim_name]
        print(f"\n{dim_name}:")
        print(f"  Mean:     {s['mean']:8.3f}")
        print(f"  Std:      {s['std']:8.3f}")
        print(f"  Min:      {s['min']:8.3f}")
        print(f"  Max:      {s['max']:8.3f}")
        print(f"  Range:    {s['range']:8.3f}")
        print(f"  Coverage: {s['coverage']:8.3f} ({s['coverage']*100:.1f}% of expected range)")
        print(f"  KS Test:  statistic={s['ks_statistic']:.4f}, p-value={s['ks_p_value']:.4f}")
        print(f"  Uniform:  {'✓ YES' if s['is_uniform'] else '✗ NO'} (p > 0.05)")
    
    print("\n" + "-"*80)
    print("SPATIAL DISTRIBUTION METRICS")
    print("-"*80)
    
    print("\nAgent Positions:")
    agent_stats = spatial_stats['agent']
    print(f"  Coverage Area:     {agent_stats['coverage_area']:.2f} (out of {512*512:.0f})")
    print(f"  Coverage Ratio:    {agent_stats['coverage_ratio']:.4f} ({agent_stats['coverage_ratio']*100:.1f}%)")
    print(f"  X Range:           {agent_stats['x_range']:.2f}")
    print(f"  Y Range:           {agent_stats['y_range']:.2f}")
    print(f"  Mean Pairwise Dist: {agent_stats['mean_pairwise_distance']:.2f}")
    print(f"  Mean NN Distance:   {agent_stats['mean_nn_distance']:.2f}")
    
    print("\nBlock Positions:")
    block_stats = spatial_stats['block']
    print(f"  Coverage Area:     {block_stats['coverage_area']:.2f} (out of {512*512:.0f})")
    print(f"  Coverage Ratio:    {block_stats['coverage_ratio']:.4f} ({block_stats['coverage_ratio']*100:.1f}%)")
    print(f"  X Range:           {block_stats['x_range']:.2f}")
    print(f"  Y Range:           {block_stats['y_range']:.2f}")
    print(f"  Mean Pairwise Dist: {block_stats['mean_pairwise_distance']:.2f}")
    print(f"  Mean NN Distance:   {block_stats['mean_nn_distance']:.2f}")
    
    # Goal position statistics
    if goal_stats_dict is not None and goal_spatial_stats is not None:
        print("\n" + "-"*80)
        print("GOAL POSITION STATISTICS")
        print("-"*80)
        
        dim_names_goal = ['goal_x', 'goal_y', 'goal_angle']
        for dim_name in dim_names_goal:
            s = goal_stats_dict[dim_name]
            print(f"\n{dim_name}:")
            print(f"  Mean:     {s['mean']:8.3f}")
            print(f"  Std:      {s['std']:8.3f}")
            print(f"  Min:      {s['min']:8.3f}")
            print(f"  Max:      {s['max']:8.3f}")
            print(f"  Range:    {s['range']:8.3f}")
            print(f"  Coverage: {s['coverage']:8.3f} ({s['coverage']*100:.1f}% of expected range)")
            print(f"  KS Test:  statistic={s['ks_statistic']:.4f}, p-value={s['ks_p_value']:.4f}")
            print(f"  Uniform:  {'✓ YES' if s['is_uniform'] else '✗ NO'} (p > 0.05)")
        
        print("\nGoal Position Spatial Distribution:")
        print(f"  Coverage Area:     {goal_spatial_stats['coverage_area']:.2f} (out of {512*512:.0f})")
        print(f"  Coverage Ratio:    {goal_spatial_stats['coverage_ratio']:.4f} ({goal_spatial_stats['coverage_ratio']*100:.1f}%)")
        print(f"  X Range:           {goal_spatial_stats['x_range']:.2f}")
        print(f"  Y Range:           {goal_spatial_stats['y_range']:.2f}")
        print(f"  Mean Pairwise Dist: {goal_spatial_stats['mean_pairwise_distance']:.2f}")
        print(f"  Mean NN Distance:   {goal_spatial_stats['mean_nn_distance']:.2f}")
    
    print("\n" + "-"*80)
    print("RANDOMIZATION ASSESSMENT")
    print("-"*80)
    
    # Overall assessment
    all_uniform = all(stats_dict[dim]['is_uniform'] for dim in dim_names)
    good_coverage = (agent_stats['coverage_ratio'] > 0.8 and 
                    block_stats['coverage_ratio'] > 0.8)
    
    print(f"\nUniform Distribution Test: {'✓ PASS' if all_uniform else '✗ FAIL'}")
    print(f"  All dimensions pass KS test for uniformity: {all_uniform}")
    
    print(f"\nSpatial Coverage: {'✓ GOOD' if good_coverage else '⚠ LIMITED'}")
    print(f"  Agent coverage: {agent_stats['coverage_ratio']*100:.1f}%")
    print(f"  Block coverage: {block_stats['coverage_ratio']*100:.1f}%")
    
    # Goal position assessment
    goal_all_uniform = None
    goal_good_coverage = None
    if goal_stats_dict is not None and goal_spatial_stats is not None:
        dim_names_goal = ['goal_x', 'goal_y', 'goal_angle']
        goal_all_uniform = all(goal_stats_dict[dim]['is_uniform'] for dim in dim_names_goal)
        goal_good_coverage = goal_spatial_stats['coverage_ratio'] > 0.8
        
        print(f"\nGoal Position Uniform Distribution Test: {'✓ PASS' if goal_all_uniform else '✗ FAIL'}")
        print(f"  All goal dimensions pass KS test for uniformity: {goal_all_uniform}")
        
        print(f"\nGoal Position Spatial Coverage: {'✓ GOOD' if goal_good_coverage else '⚠ LIMITED'}")
        print(f"  Goal coverage: {goal_spatial_stats['coverage_ratio']*100:.1f}%")
    
    overall_assessment = "GOOD" if (all_uniform and good_coverage) else "NEEDS IMPROVEMENT"
    if goal_all_uniform is not None:
        overall_assessment = "GOOD" if (all_uniform and good_coverage and goal_all_uniform and goal_good_coverage) else "NEEDS IMPROVEMENT"
    
    print(f"\n{'='*80}")
    print(f"OVERALL ASSESSMENT: {overall_assessment}")
    print(f"{'='*80}")
    
    if overall_assessment == "GOOD":
        print("✓ The dataset shows good randomization of initial states and goal positions.")
        print("  - All dimensions appear to be uniformly distributed")
        print("  - Spatial coverage is good (>80% of the space)")
        if goal_all_uniform is not None:
            print("  - Goal positions are well randomized")
    else:
        print("⚠ The dataset may have limited randomization:")
        if not all_uniform:
            print("  - Some initial state dimensions do not follow uniform distribution")
        if not good_coverage:
            print("  - Initial state spatial coverage is limited (<80% of the space)")
        if goal_all_uniform is not None:
            if not goal_all_uniform:
                print("  - Some goal position dimensions do not follow uniform distribution")
            if not goal_good_coverage:
                print("  - Goal position spatial coverage is limited (<80% of the space)")
        print("  - Consider checking the data collection process")
    
    print("\n")


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load initial states
    initial_states, episode_ends = load_initial_states(
        args.dataset_path, 
        args.max_demos
    )
    
    # Compute statistics
    print("\nComputing statistics...")
    stats_dict = compute_statistics(initial_states)
    
    # Compute spatial distribution metrics
    print("Computing spatial distribution metrics...")
    spatial_stats = compute_spatial_distribution(initial_states)
    
    # Load goal positions
    goal_positions = load_goal_positions(
        args.dataset_path,
        initial_states,
        args.max_demos
    )
    
    # Compute goal statistics if available
    goal_stats_dict = None
    goal_spatial_stats = None
    if goal_positions is not None:
        print("\nComputing goal position statistics...")
        goal_stats_dict = compute_goal_statistics(goal_positions)
        goal_spatial_stats = compute_goal_spatial_distribution(goal_positions)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_distributions(initial_states, args.output_dir)
    
    # Visualize goal distributions if available
    if goal_positions is not None:
        visualize_goal_distributions(goal_positions, args.output_dir)
        visualize_initial_vs_goal(initial_states, goal_positions, args.output_dir)
    
    # Visualize initial state images
    visualize_initial_state_images(initial_states, args.output_dir, n_samples=args.n_visualization_samples)
    
    # Print report
    print_report(initial_states, stats_dict, spatial_stats, goal_stats_dict, goal_spatial_stats)
    
    # Save statistics to file
    import json
    output_stats = {
        'n_episodes': len(initial_states),
        'dimension_stats': stats_dict,
        'spatial_stats': spatial_stats
    }
    
    # Add goal statistics if available
    if goal_stats_dict is not None and goal_spatial_stats is not None:
        output_stats['goal_dimension_stats'] = goal_stats_dict
        output_stats['goal_spatial_stats'] = goal_spatial_stats
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.generic):
            # Handle all numpy scalar types
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    output_stats = convert_to_native(output_stats)
    
    stats_file = os.path.join(args.output_dir, 'initial_state_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(output_stats, f, indent=2)
    print(f"Saved statistics to: {stats_file}")
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

