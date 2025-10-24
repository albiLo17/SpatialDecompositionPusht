#!/usr/bin/env python3
"""
Quick Trajectory Clustering Analysis

A simplified version of the clustering analysis focused on key insights.
This script provides a quick overview of trajectory clustering patterns.

Usage:
    python quick_clustering_analysis.py
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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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


def load_and_transform_data():
    """Load dataset and apply transformations."""
    print("Loading dataset...")
    dataset_path = os.path.join(REPO_ROOT, "datasets", "pusht_toy_dataset_segmented.npz")
    dataset = ToyDataset.from_file(dataset_path)
    print(f"✓ Loaded {len(dataset)} trajectory segments")
    
    # Apply transformations
    print("Applying coordinate frame transformations...")
    transformed_datasets = {}
    
    for reference_frame in [EndEffectorPoseFrame.FIRST, EndEffectorPoseFrame.LAST]:
        # Deep copy and transform
        local_dataset = dataset.__class__(
            trajectories=dataset.trajectories.copy(),
            trajectory_labels_numerical=dataset.trajectory_labels_numerical,
            label_list=dataset.label_list
        )
        
        # Select reference position
        batched_local_pose = {
            EndEffectorPoseFrame.FIRST: local_dataset.trajectories[..., 0, :],
            EndEffectorPoseFrame.LAST: local_dataset.trajectories[..., -1, :],
        }[reference_frame]
        
        # Transform to local coordinates
        batched_local_pose = batched_local_pose[..., None, :]
        local_dataset.trajectories = -batched_local_pose + local_dataset.trajectories
        
        transformed_datasets[reference_frame.name] = local_dataset
    
    print(f"✓ Created {len(transformed_datasets)} transformed datasets")
    return dataset, transformed_datasets


def extract_simple_features(trajectories):
    """Extract simple features from trajectories."""
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
    
    features = np.column_stack([
        start_pos[:, 0], start_pos[:, 1],      # start x, y
        end_pos[:, 0], end_pos[:, 1],          # end x, y
        displacement_magnitude,               # displacement magnitude
        displacement_angle,                     # displacement angle
        traj_lengths,                          # trajectory length
        traj_ranges_x, traj_ranges_y           # spatial ranges
    ])
    
    return features


def perform_clustering_analysis(features, method_name, **kwargs):
    """Perform clustering with a specific method."""
    print(f"  - Running {method_name}...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if method_name == "K-Means":
        n_clusters = kwargs.get('n_clusters', 5)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(features_scaled)
        
    elif method_name == "GMM":
        n_components = kwargs.get('n_components', 5)
        model = GaussianMixture(n_components=n_components, random_state=42)
        labels = model.fit_predict(features_scaled)
        
    elif method_name == "HDBScan":
        min_cluster_size = kwargs.get('min_cluster_size', 20)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
        labels = model.fit_predict(features_scaled)
        
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Compute silhouette score
    if len(set(labels)) > 1:
        silhouette = silhouette_score(features_scaled, labels)
    else:
        silhouette = 0.0
    
    return {
        'model': model,
        'labels': labels,
        'silhouette': silhouette,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
    }


def visualize_clustering_results(features, results, frame_type, method_name, save_dir="clustering_results"):
    """Visualize clustering results."""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Reduce to 2D for visualization
    pca_2d = PCA(n_components=2)
    features_2d = pca_2d.fit_transform(StandardScaler().fit_transform(features))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA visualization
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=results['labels'], cmap='tab10', alpha=0.6, s=20)
    ax1.set_title(f'{method_name} Clustering\n{frame_type} Frame\nSilhouette: {results["silhouette"]:.3f}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax1)
    
    # Cluster size distribution
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    bars = ax2.bar(range(len(unique_labels)), counts, color=colors, alpha=0.7)
    ax2.set_title(f'Cluster Sizes\n{len(unique_labels)} clusters')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Trajectories')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.suptitle(f'Clustering Analysis: {frame_type} Frame', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filename = f"{save_dir}/{method_name.lower().replace('-', '_')}_{frame_type.lower()}_frame.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    
    plt.show()


def main():
    """Main analysis function."""
    print("Quick Trajectory Clustering Analysis")
    print("=" * 50)
    
    # Load and transform data
    dataset, transformed_datasets = load_and_transform_data()
    
    # Analyze both frame types
    for frame_type in ['FIRST', 'LAST']:
        print(f"\n{'='*60}")
        print(f"ANALYZING {frame_type} FRAME")
        print(f"{'='*60}")
        
        # Get trajectories
        trajectories = transformed_datasets[frame_type].trajectories
        print(f"Trajectory shape: {trajectories.shape}")
        
        # Extract features
        features = extract_simple_features(trajectories)
        print(f"Feature shape: {features.shape}")
        
        # Test different clustering methods
        methods = [
            ('K-Means', {'n_clusters': 5}),
            ('K-Means', {'n_clusters': 8}),
            ('GMM', {'n_components': 5}),
            ('GMM', {'n_components': 8}),
            ('HDBScan', {'min_cluster_size': 20}),
            ('HDBScan', {'min_cluster_size': 50})
        ]
        
        best_result = None
        best_silhouette = -1
        
        for method_name, params in methods:
            try:
                result = perform_clustering_analysis(features, method_name, **params)
                print(f"  {method_name} {params}: {result['n_clusters']} clusters, "
                      f"Silhouette: {result['silhouette']:.3f}")
                
                if result['silhouette'] > best_silhouette:
                    best_silhouette = result['silhouette']
                    best_result = (method_name, params, result)
                    
            except Exception as e:
                print(f"  {method_name} {params}: Error - {e}")
        
        # Visualize best result
        if best_result:
            method_name, params, result = best_result
            print(f"\nBest result: {method_name} {params} (Silhouette: {result['silhouette']:.3f})")
            visualize_clustering_results(features, result, frame_type, 
                                       f"{method_name} {params}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print("Check the 'clustering_results/' directory for saved plots.")


if __name__ == "__main__":
    main()
