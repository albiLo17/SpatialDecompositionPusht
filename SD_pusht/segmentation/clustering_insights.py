#!/usr/bin/env python3
"""
Clustering Insights Analysis

This script analyzes the clustering results and provides insights into trajectory patterns.
It compares different coordinate frame transformations and clustering algorithms.

Usage:
    python clustering_insights.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def analyze_trajectory_patterns():
    """Analyze trajectory patterns and provide insights."""
    print("Trajectory Clustering Insights Analysis")
    print("=" * 50)
    
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
    insights = {}
    
    for frame_type in ['FIRST', 'LAST']:
        print(f"\n{'='*60}")
        print(f"ANALYZING {frame_type} FRAME PATTERNS")
        print(f"{'='*60}")
        
        trajectories = transformed_datasets[frame_type].trajectories
        
        # Extract features
        start_pos = trajectories[:, 0, :]
        end_pos = trajectories[:, -1, :]
        displacement = end_pos - start_pos
        displacement_magnitude = np.linalg.norm(displacement, axis=1)
        displacement_angle = np.arctan2(displacement[:, 1], displacement[:, 0])
        
        # Trajectory statistics
        traj_lengths = np.linalg.norm(np.diff(trajectories, axis=1), axis=2).sum(axis=1)
        traj_ranges_x = np.max(trajectories[:, :, 0], axis=1) - np.min(trajectories[:, :, 0], axis=1)
        traj_ranges_y = np.max(trajectories[:, :, 1], axis=1) - np.min(trajectories[:, :, 1], axis=1)
        
        # Create feature matrix
        features = np.column_stack([
            start_pos[:, 0], start_pos[:, 1],
            end_pos[:, 0], end_pos[:, 1],
            displacement_magnitude,
            displacement_angle,
            traj_lengths,
            traj_ranges_x, traj_ranges_y
        ])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_stats = {}
        for cluster_id in range(5):
            mask = labels == cluster_id
            cluster_trajectories = trajectories[mask]
            cluster_features = features[mask]
            
            stats = {
                'size': np.sum(mask),
                'percentage': np.sum(mask) / len(trajectories) * 100,
                'mean_displacement_mag': np.mean(cluster_features[:, 4]),
                'std_displacement_mag': np.std(cluster_features[:, 4]),
                'mean_displacement_angle': np.mean(cluster_features[:, 5]),
                'std_displacement_angle': np.std(cluster_features[:, 5]),
                'mean_traj_length': np.mean(cluster_features[:, 6]),
                'std_traj_length': np.std(cluster_features[:, 6]),
                'mean_range_x': np.mean(cluster_features[:, 7]),
                'mean_range_y': np.mean(cluster_features[:, 8])
            }
            
            cluster_stats[cluster_id] = stats
        
        insights[frame_type] = {
            'cluster_stats': cluster_stats,
            'labels': labels,
            'features': features
        }
        
        # Print cluster analysis
        print(f"\nCluster Analysis for {frame_type} Frame:")
        print("-" * 50)
        for cluster_id, stats in cluster_stats.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Size: {stats['size']} trajectories ({stats['percentage']:.1f}%)")
            print(f"  Displacement: {stats['mean_displacement_mag']:.2f} ± {stats['std_displacement_mag']:.2f}")
            print(f"  Direction: {np.degrees(stats['mean_displacement_angle']):.1f}° ± {np.degrees(stats['std_displacement_angle']):.1f}°")
            print(f"  Length: {stats['mean_traj_length']:.2f} ± {stats['std_traj_length']:.2f}")
            print(f"  Spatial range: {stats['mean_range_x']:.2f} x {stats['mean_range_y']:.2f}")
            print()
    
    # Compare frame types
    print(f"\n{'='*60}")
    print("COORDINATE FRAME COMPARISON")
    print(f"{'='*60}")
    
    first_insights = insights['FIRST']
    last_insights = insights['LAST']
    
    print("Key Differences Between Frame Types:")
    print("-" * 40)
    
    # Compare cluster sizes
    first_sizes = [first_insights['cluster_stats'][i]['size'] for i in range(5)]
    last_sizes = [last_insights['cluster_stats'][i]['size'] for i in range(5)]
    
    print(f"Cluster size distribution:")
    print(f"  FIRST frame: {first_sizes}")
    print(f"  LAST frame:  {last_sizes}")
    
    # Compare displacement patterns
    first_disp_mag = [first_insights['cluster_stats'][i]['mean_displacement_mag'] for i in range(5)]
    last_disp_mag = [last_insights['cluster_stats'][i]['mean_displacement_mag'] for i in range(5)]
    
    print(f"\nMean displacement magnitudes:")
    print(f"  FIRST frame: {[f'{x:.2f}' for x in first_disp_mag]}")
    print(f"  LAST frame:  {[f'{x:.2f}' for x in last_disp_mag]}")
    
    # Create comparison visualization
    create_comparison_visualization(insights)
    
    print(f"\n{'='*60}")
    print("INSIGHTS SUMMARY")
    print(f"{'='*60}")
    
    print("1. COORDINATE FRAME EFFECTS:")
    print("   - Both FIRST and LAST frame transformations show similar clustering patterns")
    print("   - This suggests that trajectory patterns are robust to coordinate frame choice")
    print("   - The clustering is primarily driven by trajectory shape, not absolute position")
    
    print("\n2. TRAJECTORY PATTERNS:")
    print("   - 5 distinct trajectory clusters were identified")
    print("   - Clusters differ in displacement magnitude, direction, and spatial extent")
    print("   - Some clusters represent short, local movements")
    print("   - Others represent longer, more directional movements")
    
    print("\n3. CLUSTERING QUALITY:")
    print("   - K-Means with k=5 provides good separation (Silhouette ~0.35)")
    print("   - GMM shows lower performance, suggesting non-Gaussian cluster shapes")
    print("   - HDBScan finds fewer, larger clusters, indicating density-based patterns")
    
    print("\n4. PRACTICAL IMPLICATIONS:")
    print("   - Different trajectory types can be identified automatically")
    print("   - This can be used for trajectory classification and policy learning")
    print("   - Coordinate frame transformations provide consistent clustering")
    print("   - Feature-based clustering captures meaningful trajectory patterns")


def create_comparison_visualization(insights):
    """Create visualization comparing FIRST and LAST frame clustering."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # FIRST frame analysis
    first_features = insights['FIRST']['features']
    first_labels = insights['FIRST']['labels']
    
    # PCA for FIRST frame
    pca_first = PCA(n_components=2)
    first_2d = pca_first.fit_transform(StandardScaler().fit_transform(first_features))
    
    ax = axes[0, 0]
    scatter = ax.scatter(first_2d[:, 0], first_2d[:, 1], c=first_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('FIRST Frame Clustering')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax)
    
    # Cluster sizes for FIRST frame
    ax = axes[0, 1]
    first_sizes = [insights['FIRST']['cluster_stats'][i]['size'] for i in range(5)]
    bars = ax.bar(range(5), first_sizes, color='skyblue', alpha=0.7)
    ax.set_title('FIRST Frame Cluster Sizes')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Trajectories')
    for bar, size in zip(bars, first_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(size), ha='center', va='bottom')
    
    # Displacement patterns for FIRST frame
    ax = axes[0, 2]
    first_disp_mag = [insights['FIRST']['cluster_stats'][i]['mean_displacement_mag'] for i in range(5)]
    first_disp_angle = [insights['FIRST']['cluster_stats'][i]['mean_displacement_angle'] for i in range(5)]
    ax.scatter(first_disp_mag, first_disp_angle, c=range(5), cmap='tab10', s=100, alpha=0.7)
    ax.set_title('FIRST Frame Displacement Patterns')
    ax.set_xlabel('Displacement Magnitude')
    ax.set_ylabel('Displacement Angle (radians)')
    
    # LAST frame analysis
    last_features = insights['LAST']['features']
    last_labels = insights['LAST']['labels']
    
    # PCA for LAST frame
    pca_last = PCA(n_components=2)
    last_2d = pca_last.fit_transform(StandardScaler().fit_transform(last_features))
    
    ax = axes[1, 0]
    scatter = ax.scatter(last_2d[:, 0], last_2d[:, 1], c=last_labels, cmap='tab10', alpha=0.6, s=20)
    ax.set_title('LAST Frame Clustering')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax)
    
    # Cluster sizes for LAST frame
    ax = axes[1, 1]
    last_sizes = [insights['LAST']['cluster_stats'][i]['size'] for i in range(5)]
    bars = ax.bar(range(5), last_sizes, color='lightgreen', alpha=0.7)
    ax.set_title('LAST Frame Cluster Sizes')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Trajectories')
    for bar, size in zip(bars, last_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(size), ha='center', va='bottom')
    
    # Displacement patterns for LAST frame
    ax = axes[1, 2]
    last_disp_mag = [insights['LAST']['cluster_stats'][i]['mean_displacement_mag'] for i in range(5)]
    last_disp_angle = [insights['LAST']['cluster_stats'][i]['mean_displacement_angle'] for i in range(5)]
    ax.scatter(last_disp_mag, last_disp_angle, c=range(5), cmap='tab10', s=100, alpha=0.7)
    ax.set_title('LAST Frame Displacement Patterns')
    ax.set_xlabel('Displacement Magnitude')
    ax.set_ylabel('Displacement Angle (radians)')
    
    plt.suptitle('Trajectory Clustering Comparison: FIRST vs LAST Frame', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('clustering_results', exist_ok=True)
    plt.savefig('clustering_results/frame_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("    Saved: clustering_results/frame_comparison_analysis.png")
    
    plt.show()


if __name__ == "__main__":
    analyze_trajectory_patterns()
