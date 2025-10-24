#!/usr/bin/env python3
"""
HDBSCAN Hyperparameter Tuning

This script demonstrates how to properly tune HDBSCAN hyperparameters
for trajectory clustering analysis.

Usage:
    python hdbscan_tuning.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan

# Add paths for imports
REPO_ROOT = "/home/albi/Workspace/SpatialDecompositionPusht"
SPATIALDECOMP_PATH = os.path.join(REPO_ROOT, "spatialdecomposition")
if SPATIALDECOMP_PATH not in sys.path:
    sys.path.append(SPATIALDECOMP_PATH)

sys.path.extend([
    "/home/albi/Workspace/SpatialDecompositionPusht/libero",
    "/home/albi/Workspace/SpatialDecompositionPusht/robomimic", 
    "/home/albi/Workspace/SpatialDecompositionPusht/robosuite",
    "/home/albi/Workspace/SpatialDecompositionPusht/spatialdecomposition",
    "/home/albi/Workspace/SpatialDecompositionPusht/diffusion_policy",
])

from SpatialDecomposition.TwoD_table_play.data import ToyDataset
from SpatialDecomposition.datasets import EndEffectorPoseFrame


def extract_features(trajectories):
    """Extract features from trajectories."""
    # Start and end positions
    start_pos = trajectories[:, 0, :]  # [N, 2]
    end_pos = trajectories[:, -1, :]  # [N, 2]
    
    # Displacement vector
    displacement = end_pos - start_pos  # [N, 2]
    displacement_magnitude = np.linalg.norm(displacement, axis=1)  # [N]
    displacement_angle = np.arctan2(displacement[:, 1], displacement[:, 0])  # [N]
    
    # Trajectory length
    traj_lengths = np.linalg.norm(np.diff(trajectories, axis=1), axis=2).sum(axis=1)  # [N]
    
    # Spatial range
    traj_ranges_x = np.max(trajectories[:, :, 0], axis=1) - np.min(trajectories[:, :, 0], axis=1)
    traj_ranges_y = np.max(trajectories[:, :, 1], axis=1) - np.min(trajectories[:, :, 1], axis=1)
    
    # Velocity statistics
    velocities = np.diff(trajectories, axis=1)  # [N, T-1, 2]
    vel_magnitudes = np.linalg.norm(velocities, axis=2)  # [N, T-1]
    mean_vel = np.mean(vel_magnitudes, axis=1)  # [N]
    max_vel = np.max(vel_magnitudes, axis=1)  # [N]
    vel_std = np.std(vel_magnitudes, axis=1)  # [N]
    
    features = np.column_stack([
        start_pos[:, 0], start_pos[:, 1],      # start x, y
        end_pos[:, 0], end_pos[:, 1],          # end x, y
        displacement_magnitude,               # displacement magnitude
        displacement_angle,                     # displacement angle
        traj_lengths,                          # trajectory length
        traj_ranges_x, traj_ranges_y,          # spatial ranges
        mean_vel, max_vel, vel_std             # velocity statistics
    ])
    
    return features


def tune_hdbscan_hyperparameters(features, frame_type):
    """Tune HDBSCAN hyperparameters systematically."""
    print(f"\n{'='*60}")
    print(f"HDBSCAN HYPERPARAMETER TUNING: {frame_type} Frame")
    print(f"{'='*60}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Define parameter grids
    min_cluster_sizes = [5, 10, 20, 50, 100, 200]
    min_samples_list = [3, 5, 10, 15, 20]
    cluster_selection_epsilons = [0.0, 0.1, 0.2, 0.3, 0.5]
    metrics = ['euclidean', 'manhattan']  # Removed 'cosine' as it's not supported
    
    results = []
    
    print("Testing parameter combinations...")
    print("Format: min_cluster_size, min_samples, epsilon, metric -> n_clusters, silhouette, noise_%")
    print("-" * 80)
    
    best_result = None
    best_silhouette = -1
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            # Skip if min_samples >= min_cluster_size (invalid)
            if min_samples >= min_cluster_size:
                continue
                
            for epsilon in cluster_selection_epsilons:
                for metric in metrics:
                    try:
                        # Create HDBSCAN model
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=epsilon,
                            metric=metric,
                            cluster_selection_method='eom'
                        )
                        
                        # Fit and predict
                        labels = clusterer.fit_predict(features_scaled)
                        
                        # Analyze results
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        noise_percentage = (n_noise / len(labels)) * 100
                        
                        # Compute metrics
                        if n_clusters > 1:
                            silhouette = silhouette_score(features_scaled, labels)
                            calinski = calinski_harabasz_score(features_scaled, labels)
                            davies_bouldin = davies_bouldin_score(features_scaled, labels)
                        else:
                            silhouette = calinski = davies_bouldin = 0.0
                        
                        result = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'epsilon': epsilon,
                            'metric': metric,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'noise_percentage': noise_percentage,
                            'silhouette': silhouette,
                            'calinski_harabasz': calinski,
                            'davies_bouldin': davies_bouldin,
                            'labels': labels
                        }
                        
                        results.append(result)
                        
                        # Print result
                        print(f"{min_cluster_size:3d}, {min_samples:2d}, {epsilon:3.1f}, {metric:10s} -> "
                              f"{n_clusters:2d} clusters, {silhouette:6.3f}, {noise_percentage:5.1f}% noise")
                        
                        # Track best result
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_result = result
                            
                    except Exception as e:
                        print(f"{min_cluster_size:3d}, {min_samples:2d}, {epsilon:3.1f}, {metric:10s} -> ERROR: {e}")
    
    return results, best_result


def analyze_parameter_effects(results, frame_type):
    """Analyze the effects of different parameters."""
    print(f"\n{'='*60}")
    print(f"PARAMETER EFFECT ANALYSIS: {frame_type} Frame")
    print(f"{'='*60}")
    
    # Convert to numpy arrays for analysis
    min_cluster_sizes = np.array([r['min_cluster_size'] for r in results])
    min_samples = np.array([r['min_samples'] for r in results])
    epsilons = np.array([r['epsilon'] for r in results])
    metrics = [r['metric'] for r in results]
    n_clusters = np.array([r['n_clusters'] for r in results])
    silhouettes = np.array([r['silhouette'] for r in results])
    noise_percentages = np.array([r['noise_percentage'] for r in results])
    
    # Analyze min_cluster_size effects
    print("\n1. MIN_CLUSTER_SIZE EFFECTS:")
    print("-" * 40)
    for size in sorted(set(min_cluster_sizes)):
        mask = min_cluster_sizes == size
        if np.any(mask):
            avg_silhouette = np.mean(silhouettes[mask])
            avg_clusters = np.mean(n_clusters[mask])
            avg_noise = np.mean(noise_percentages[mask])
            print(f"  Size {size:3d}: {avg_clusters:5.1f} clusters, {avg_silhouette:6.3f} silhouette, {avg_noise:5.1f}% noise")
    
    # Analyze min_samples effects
    print("\n2. MIN_SAMPLES EFFECTS:")
    print("-" * 40)
    for samples in sorted(set(min_samples)):
        mask = min_samples == samples
        if np.any(mask):
            avg_silhouette = np.mean(silhouettes[mask])
            avg_clusters = np.mean(n_clusters[mask])
            avg_noise = np.mean(noise_percentages[mask])
            print(f"  Samples {samples:2d}: {avg_clusters:5.1f} clusters, {avg_silhouette:6.3f} silhouette, {avg_noise:5.1f}% noise")
    
    # Analyze epsilon effects
    print("\n3. EPSILON EFFECTS:")
    print("-" * 40)
    for epsilon in sorted(set(epsilons)):
        mask = epsilons == epsilon
        if np.any(mask):
            avg_silhouette = np.mean(silhouettes[mask])
            avg_clusters = np.mean(n_clusters[mask])
            avg_noise = np.mean(noise_percentages[mask])
            print(f"  Epsilon {epsilon:3.1f}: {avg_clusters:5.1f} clusters, {avg_silhouette:6.3f} silhouette, {avg_noise:5.1f}% noise")
    
    # Analyze metric effects
    print("\n4. METRIC EFFECTS:")
    print("-" * 40)
    for metric in sorted(set(metrics)):
        mask = np.array([m == metric for m in metrics])
        if np.any(mask):
            avg_silhouette = np.mean(silhouettes[mask])
            avg_clusters = np.mean(n_clusters[mask])
            avg_noise = np.mean(noise_percentages[mask])
            print(f"  {metric:10s}: {avg_clusters:5.1f} clusters, {avg_silhouette:6.3f} silhouette, {avg_noise:5.1f}% noise")


def visualize_parameter_effects(results, frame_type, save_dir="clustering_results"):
    """Create visualizations of parameter effects."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    min_cluster_sizes = np.array([r['min_cluster_size'] for r in results])
    min_samples = np.array([r['min_samples'] for r in results])
    epsilons = np.array([r['epsilon'] for r in results])
    n_clusters = np.array([r['n_clusters'] for r in results])
    silhouettes = np.array([r['silhouette'] for r in results])
    noise_percentages = np.array([r['noise_percentage'] for r in results])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Min cluster size vs silhouette
    ax = axes[0, 0]
    for size in sorted(set(min_cluster_sizes)):
        mask = min_cluster_sizes == size
        if np.any(mask):
            ax.scatter([size] * np.sum(mask), silhouettes[mask], alpha=0.6, s=30, label=f'Size {size}')
    ax.set_xlabel('Min Cluster Size')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Min Cluster Size vs Silhouette Score')
    ax.grid(True, alpha=0.3)
    
    # 2. Min samples vs silhouette
    ax = axes[0, 1]
    for samples in sorted(set(min_samples)):
        mask = min_samples == samples
        if np.any(mask):
            ax.scatter([samples] * np.sum(mask), silhouettes[mask], alpha=0.6, s=30, label=f'Samples {samples}')
    ax.set_xlabel('Min Samples')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Min Samples vs Silhouette Score')
    ax.grid(True, alpha=0.3)
    
    # 3. Epsilon vs silhouette
    ax = axes[1, 0]
    for epsilon in sorted(set(epsilons)):
        mask = epsilons == epsilon
        if np.any(mask):
            ax.scatter([epsilon] * np.sum(mask), silhouettes[mask], alpha=0.6, s=30, label=f'Epsilon {epsilon}')
    ax.set_xlabel('Cluster Selection Epsilon')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Epsilon vs Silhouette Score')
    ax.grid(True, alpha=0.3)
    
    # 4. Noise percentage vs silhouette
    ax = axes[1, 1]
    scatter = ax.scatter(noise_percentages, silhouettes, c=n_clusters, cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel('Noise Percentage')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Noise Percentage vs Silhouette Score')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Number of Clusters')
    
    plt.suptitle(f'HDBSCAN Parameter Effects: {frame_type} Frame', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filename = f"{save_dir}/hdbscan_parameter_effects_{frame_type.lower()}_frame.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    
    plt.show()


def main():
    """Main function for HDBSCAN hyperparameter tuning."""
    print("HDBSCAN Hyperparameter Tuning for Trajectory Clustering")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    dataset_path = os.path.join(REPO_ROOT, "datasets", "pusht_toy_dataset_segmented.npz")
    dataset = ToyDataset.from_file(dataset_path)
    print(f"✓ Loaded {len(dataset)} trajectory segments")
    
    # Apply transformations
    print("Applying coordinate frame transformations...")
    transformed_datasets = {}
    
    for reference_frame in [EndEffectorPoseFrame.FIRST, EndEffectorPoseFrame.LAST]:
        local_dataset = dataset.__class__(
            trajectories=dataset.trajectories.copy(),
            trajectory_labels_numerical=dataset.trajectory_labels_numerical,
            label_list=dataset.label_list
        )
        
        batched_local_pose = {
            EndEffectorPoseFrame.FIRST: local_dataset.trajectories[..., 0, :],
            EndEffectorPoseFrame.LAST: local_dataset.trajectories[..., -1, :],
        }[reference_frame]
        
        batched_local_pose = batched_local_pose[..., None, :]
        local_dataset.trajectories = -batched_local_pose + local_dataset.trajectories
        transformed_datasets[reference_frame.name] = local_dataset
    
    # Analyze both frame types
    for frame_type in ['FIRST', 'LAST']:
        print(f"\n{'='*80}")
        print(f"ANALYZING {frame_type} FRAME")
        print(f"{'='*80}")
        
        # Get trajectories and extract features
        trajectories = transformed_datasets[frame_type].trajectories
        features = extract_features(trajectories)
        print(f"Feature shape: {features.shape}")
        
        # Tune HDBSCAN hyperparameters
        results, best_result = tune_hdbscan_hyperparameters(features, frame_type)
        
        # Analyze parameter effects
        analyze_parameter_effects(results, frame_type)
        
        # Create visualizations
        visualize_parameter_effects(results, frame_type)
        
        # Print best result
        if best_result:
            print(f"\n{'='*60}")
            print(f"BEST HDBSCAN RESULT: {frame_type} Frame")
            print(f"{'='*60}")
            print(f"Parameters:")
            print(f"  min_cluster_size: {best_result['min_cluster_size']}")
            print(f"  min_samples: {best_result['min_samples']}")
            print(f"  cluster_selection_epsilon: {best_result['epsilon']}")
            print(f"  metric: {best_result['metric']}")
            print(f"Results:")
            print(f"  n_clusters: {best_result['n_clusters']}")
            print(f"  n_noise: {best_result['n_noise']} ({best_result['noise_percentage']:.1f}%)")
            print(f"  silhouette: {best_result['silhouette']:.3f}")
            print(f"  calinski_harabasz: {best_result['calinski_harabasz']:.1f}")
            print(f"  davies_bouldin: {best_result['davies_bouldin']:.3f}")
    
    print(f"\n{'='*80}")
    print("HDBSCAN HYPERPARAMETER TUNING COMPLETE!")
    print(f"{'='*80}")
    print("Key insights:")
    print("1. min_cluster_size: Controls cluster size - larger values → fewer, larger clusters")
    print("2. min_samples: Controls density requirements - larger values → more selective clustering")
    print("3. cluster_selection_epsilon: Controls cluster merging - larger values → more merging")
    print("4. metric: Distance metric - 'euclidean' for most cases, 'cosine' for high-dim data")
    print("\nRecommendations:")
    print("- Start with min_cluster_size = 10-20% of dataset size")
    print("- Set min_samples = 1/3 to 1/2 of min_cluster_size")
    print("- Use epsilon = 0.0 for automatic selection, or 0.1-0.3 for more merging")
    print("- Use 'euclidean' metric for most trajectory clustering tasks")


if __name__ == "__main__":
    main()
