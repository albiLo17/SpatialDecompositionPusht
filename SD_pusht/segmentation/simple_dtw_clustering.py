#!/usr/bin/env python3
"""
Simple DTW-based Trajectory Clustering

A simplified version of DTW clustering that works reliably.
This script demonstrates DTW-based clustering without complex parameter tuning.

Usage:
    python simple_dtw_clustering.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
import hdbscan

# DTW distance
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    print("Warning: dtaidistance not available. Install with: pip install dtaidistance")

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


def compute_dtw_distance_matrix(trajectories, max_samples=100):
    """
    Compute DTW distance matrix for trajectories.
    
    Args:
        trajectories: Array of trajectories [N, T, D]
        max_samples: Maximum number of samples to use
        
    Returns:
        DTW distance matrix
    """
    n_trajectories = len(trajectories)
    
    # Limit number of samples for computational efficiency
    if n_trajectories > max_samples:
        indices = np.random.choice(n_trajectories, max_samples, replace=False)
        trajectories = trajectories[indices]
        n_trajectories = max_samples
        print(f"Using {n_trajectories} sampled trajectories")
    
    print(f"Computing DTW distances for {n_trajectories} trajectories...")
    print("This may take a while...")
    
    # Initialize distance matrix
    dtw_matrix = np.zeros((n_trajectories, n_trajectories))
    
    # Compute DTW distances
    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):
            # Compute DTW distance between trajectories i and j
            try:
                distance = dtw.distance(trajectories[i], trajectories[j])
                dtw_matrix[i, j] = distance
                dtw_matrix[j, i] = distance  # Symmetric matrix
            except Exception as e:
                print(f"Warning: DTW computation failed for pair ({i}, {j}): {e}")
                dtw_matrix[i, j] = 0.0
                dtw_matrix[j, i] = 0.0
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_trajectories} trajectories")
    
    print(f"✓ DTW distance matrix computed: {dtw_matrix.shape}")
    return dtw_matrix, trajectories


def perform_dtw_clustering(dtw_matrix, n_clusters_list=[3, 5, 8]):
    """
    Perform clustering using DTW distance matrix.
    
    Args:
        dtw_matrix: DTW distance matrix
        n_clusters_list: List of cluster numbers to try
        
    Returns:
        Dictionary containing clustering results
    """
    print("Performing DTW-based clustering...")
    
    results = {}
    
    # 1. Agglomerative Clustering with different numbers of clusters
    print("  - Running Agglomerative Clustering...")
    agg_results = {}
    for n_clusters in n_clusters_list:
        try:
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = agg_clustering.fit_predict(dtw_matrix)
            
            # Compute silhouette score
            if len(set(labels)) > 1:
                silhouette = silhouette_score(dtw_matrix, labels, metric='precomputed')
            else:
                silhouette = 0.0
            
            agg_results[n_clusters] = {
                'model': agg_clustering,
                'labels': labels,
                'silhouette': silhouette,
                'n_clusters': n_clusters
            }
            
            print(f"    k={n_clusters}: {len(set(labels))} clusters, silhouette={silhouette:.3f}")
            
        except Exception as e:
            print(f"    k={n_clusters}: Error - {e}")
    
    results['agglomerative'] = agg_results
    
    # 2. HDBScan
    print("  - Running HDBScan...")
    hdbscan_results = {}
    for min_cluster_size in [5, 10, 20]:
        try:
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=3,
                metric='precomputed'
            )
            labels = hdb.fit_predict(dtw_matrix)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                silhouette = silhouette_score(dtw_matrix, labels, metric='precomputed')
            else:
                silhouette = 0.0
                
            hdbscan_results[min_cluster_size] = {
                'model': hdb,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette
            }
            
            print(f"    min_size={min_cluster_size}: {n_clusters} clusters, {n_noise} noise, silhouette={silhouette:.3f}")
            
        except Exception as e:
            print(f"    min_size={min_cluster_size}: Error - {e}")
    
    results['hdbscan'] = hdbscan_results
    
    return results


def visualize_dtw_clustering(dtw_matrix, results, trajectories, frame_type, save_dir="clustering_results"):
    """
    Visualize DTW clustering results.
    
    Args:
        dtw_matrix: DTW distance matrix
        results: Clustering results
        trajectories: Original trajectories
        frame_type: 'FIRST' or 'LAST' frame
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating DTW clustering visualizations for {frame_type} frame...")
    
    # Reduce DTW distances to 2D for visualization
    print("  - Computing 2D embedding of DTW distances...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    dtw_2d = mds.fit_transform(dtw_matrix)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. DTW distance heatmap
    ax = axes[0, 0]
    im = ax.imshow(dtw_matrix, cmap='viridis', aspect='auto')
    ax.set_title('DTW Distance Matrix')
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('Trajectory Index')
    plt.colorbar(im, ax=ax)
    
    # 2. MDS embedding
    ax = axes[0, 1]
    scatter = ax.scatter(dtw_2d[:, 0], dtw_2d[:, 1], alpha=0.6, s=20)
    ax.set_title('MDS Embedding of DTW Distances')
    ax.set_xlabel('MDS Component 1')
    ax.set_ylabel('MDS Component 2')
    
    # 3. Best Agglomerative result
    agg_results = results['agglomerative']
    if agg_results:
        best_k = max(agg_results.keys(), key=lambda k: agg_results[k]['silhouette'])
        agg_labels = agg_results[best_k]['labels']
        
        ax = axes[1, 0]
        scatter = ax.scatter(dtw_2d[:, 0], dtw_2d[:, 1], c=agg_labels, cmap='tab10', alpha=0.6, s=20)
        ax.set_title(f'Agglomerative Clustering (k={best_k})\nSilhouette: {agg_results[best_k]["silhouette"]:.3f}')
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        plt.colorbar(scatter, ax=ax)
    
    # 4. Best HDBScan result
    hdbscan_results = results['hdbscan']
    if hdbscan_results:
        best_min_size = max(hdbscan_results.keys(), key=lambda k: hdbscan_results[k]['silhouette'])
        hdbscan_labels = hdbscan_results[best_min_size]['labels']
        
        ax = axes[1, 1]
        scatter = ax.scatter(dtw_2d[:, 0], dtw_2d[:, 1], c=hdbscan_labels, cmap='tab10', alpha=0.6, s=20)
        ax.set_title(f'HDBScan Clustering (min_size={best_min_size})\nSilhouette: {hdbscan_results[best_min_size]["silhouette"]:.3f}')
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        plt.colorbar(scatter, ax=ax)
    
    plt.suptitle(f'DTW-based Clustering Analysis: {frame_type} Frame', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filename = f"{save_dir}/dtw_clustering_{frame_type.lower()}_frame.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    
    plt.show()


def main():
    """Main function for DTW-based clustering analysis."""
    print("Simple DTW-based Trajectory Clustering Analysis")
    print("=" * 60)
    
    if not DTW_AVAILABLE:
        print("❌ Error: dtaidistance package is required for DTW clustering.")
        print("Install it with: pip install dtaidistance")
        return
    
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
    
    # Analyze both frame types
    for frame_type in ['FIRST', 'LAST']:
        print(f"\n{'='*80}")
        print(f"ANALYZING {frame_type} FRAME WITH DTW DISTANCES")
        print(f"{'='*80}")
        
        # Get trajectories
        trajectories = transformed_datasets[frame_type].trajectories
        print(f"Trajectory shape: {trajectories.shape}")
        
        # Compute DTW distances (limit to 100 samples for efficiency)
        dtw_matrix, sampled_trajectories = compute_dtw_distance_matrix(trajectories, max_samples=100)
        
        # Perform clustering
        results = perform_dtw_clustering(dtw_matrix, n_clusters_list=[3, 5, 8])
        
        # Print summary
        print(f"\n--- DTW Clustering Summary: {frame_type} Frame ---")
        
        # Agglomerative results
        agg_results = results['agglomerative']
        if agg_results:
            print("Agglomerative Clustering:")
            for k, r in agg_results.items():
                print(f"  k={k}: {r['n_clusters']} clusters, silhouette={r['silhouette']:.3f}")
        
        # HDBScan results
        hdbscan_results = results['hdbscan']
        if hdbscan_results:
            print("HDBScan:")
            for min_size, r in hdbscan_results.items():
                print(f"  min_size={min_size}: {r['n_clusters']} clusters, {r['n_noise']} noise, silhouette={r['silhouette']:.3f}")
        
        # Create visualizations
        visualize_dtw_clustering(dtw_matrix, results, sampled_trajectories, frame_type)
    
    print(f"\n{'='*80}")
    print("DTW CLUSTERING ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Key advantages of DTW clustering:")
    print("1. No manual feature engineering required")
    print("2. Directly compares trajectory shapes and timing")
    print("3. Handles trajectories of different lengths")
    print("4. Robust to time warping and speed variations")
    print("5. Captures temporal patterns automatically")


if __name__ == "__main__":
    main()
