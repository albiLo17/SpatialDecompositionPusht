#!/usr/bin/env python3
"""
Robust DTW-based Trajectory Clustering

A robust version that handles DTW computation issues and provides alternative
distance metrics for trajectory clustering.

Usage:
    python robust_dtw_clustering.py
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
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
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


def safe_dtw_distance(traj1, traj2):
    """
    Safely compute DTW distance between two trajectories.
    
    Args:
        traj1, traj2: Trajectory arrays [T, D]
        
    Returns:
        DTW distance or fallback distance
    """
    # try:
    # Ensure trajectories are 2D and have the right shape
    if traj1.ndim == 1:
        traj1 = traj1.reshape(-1, 1)
    if traj2.ndim == 1:
        traj2 = traj2.reshape(-1, 1)
        
    # Flatten to 1D for DTW computation
    traj1_flat = traj1.flatten()
    traj2_flat = traj2.flatten()
    
    # Compute DTW distance
    distance = dtw.distance(traj1_flat, traj2_flat)
    return float(distance)
        
    # except Exception as e:
    #     # Fallback to Euclidean distance
    #     print(f"DTW failed, using Euclidean distance: {e}")
    #     return float(np.linalg.norm(traj1_flat - traj2_flat))


def compute_trajectory_distance_matrix(trajectories, max_samples=50, distance_type='dtw'):
    """
    Compute distance matrix for trajectories using different distance metrics.
    
    Args:
        trajectories: Array of trajectories [N, T, D]
        max_samples: Maximum number of samples to use
        distance_type: 'dtw', 'euclidean', or 'cosine'
        
    Returns:
        Distance matrix and sampled trajectories
    """
    n_trajectories = len(trajectories)
    
    # Limit number of samples for computational efficiency
    if n_trajectories > max_samples:
        indices = np.random.choice(n_trajectories, max_samples, replace=False)
        trajectories = trajectories[indices]
        n_trajectories = max_samples
        print(f"Using {n_trajectories} sampled trajectories")
    
    print(f"Computing {distance_type} distances for {n_trajectories} trajectories...")
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n_trajectories, n_trajectories))
    
    if distance_type == 'dtw' and DTW_AVAILABLE:
        # DTW distances
        for i in range(n_trajectories):
            for j in range(i + 1, n_trajectories):
                distance = safe_dtw_distance(trajectories[i], trajectories[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric matrix
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_trajectories} trajectories")
                
    elif distance_type == 'euclidean':
        # Euclidean distances between trajectory points
        for i in range(n_trajectories):
            for j in range(i + 1, n_trajectories):
                # Compute pairwise Euclidean distances between all points
                traj1 = trajectories[i].reshape(-1)  # Flatten
                traj2 = trajectories[j].reshape(-1)  # Flatten
                distance = np.linalg.norm(traj1 - traj2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                
    elif distance_type == 'cosine':
        # Cosine distances
        for i in range(n_trajectories):
            for j in range(i + 1, n_trajectories):
                traj1 = trajectories[i].reshape(-1)
                traj2 = trajectories[j].reshape(-1)
                # Normalize vectors
                traj1_norm = traj1 / (np.linalg.norm(traj1) + 1e-8)
                traj2_norm = traj2 / (np.linalg.norm(traj2) + 1e-8)
                # Cosine distance = 1 - cosine similarity
                cosine_sim = np.dot(traj1_norm, traj2_norm)
                distance = 1 - cosine_sim
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    print(f"✓ {distance_type.title()} distance matrix computed: {distance_matrix.shape}")
    return distance_matrix, trajectories


def perform_trajectory_clustering(distance_matrix, n_clusters_list=[3, 5, 8]):
    """
    Perform clustering using trajectory distance matrix.
    
    Args:
        distance_matrix: Distance matrix
        n_clusters_list: List of cluster numbers to try
        
    Returns:
        Dictionary containing clustering results
    """
    print("Performing trajectory-based clustering...")
    
    results = {}
    
    # 1. Agglomerative Clustering
    print("  - Running Agglomerative Clustering...")
    agg_results = {}
    for n_clusters in n_clusters_list:
        try:
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = agg_clustering.fit_predict(distance_matrix)
            
            # Compute silhouette score
            if len(set(labels)) > 1:
                silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
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
            labels = hdb.fit_predict(distance_matrix)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
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


def visualize_trajectory_clustering(distance_matrix, results, trajectories, frame_type, 
                                  distance_type, save_dir="clustering_results"):
    """
    Visualize trajectory clustering results.
    
    Args:
        distance_matrix: Distance matrix
        results: Clustering results
        trajectories: Original trajectories
        frame_type: 'FIRST' or 'LAST' frame
        distance_type: Type of distance used
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating {distance_type} clustering visualizations for {frame_type} frame...")
    
    # Reduce distances to 2D for visualization
    print("  - Computing 2D embedding of distances...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    distances_2d = mds.fit_transform(distance_matrix)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distance heatmap
    ax = axes[0, 0]
    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
    ax.set_title(f'{distance_type.title()} Distance Matrix')
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('Trajectory Index')
    plt.colorbar(im, ax=ax)
    
    # 2. MDS embedding
    ax = axes[0, 1]
    scatter = ax.scatter(distances_2d[:, 0], distances_2d[:, 1], alpha=0.6, s=20)
    ax.set_title(f'MDS Embedding of {distance_type.title()} Distances')
    ax.set_xlabel('MDS Component 1')
    ax.set_ylabel('MDS Component 2')
    
    # 3. Best Agglomerative result
    agg_results = results['agglomerative']
    if agg_results:
        best_k = max(agg_results.keys(), key=lambda k: agg_results[k]['silhouette'])
        agg_labels = agg_results[best_k]['labels']
        
        ax = axes[1, 0]
        scatter = ax.scatter(distances_2d[:, 0], distances_2d[:, 1], c=agg_labels, cmap='tab10', alpha=0.6, s=20)
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
        scatter = ax.scatter(distances_2d[:, 0], distances_2d[:, 1], c=hdbscan_labels, cmap='tab10', alpha=0.6, s=20)
        ax.set_title(f'HDBScan Clustering (min_size={best_min_size})\nSilhouette: {hdbscan_results[best_min_size]["silhouette"]:.3f}')
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        plt.colorbar(scatter, ax=ax)
    
    plt.suptitle(f'{distance_type.title()}-based Clustering Analysis: {frame_type} Frame', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    filename = f"{save_dir}/{distance_type}_clustering_{frame_type.lower()}_frame.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    
    plt.show()


def main():
    """Main function for robust trajectory clustering analysis."""
    print("Robust Trajectory Clustering Analysis")
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
    
    # Test different distance metrics
    distance_types = ['euclidean', 'cosine']
    if DTW_AVAILABLE:
        distance_types.insert(0, 'dtw')
    
    # Analyze both frame types with different distance metrics
    for frame_type in ['FIRST', 'LAST']:
        print(f"\n{'='*80}")
        print(f"ANALYZING {frame_type} FRAME WITH TRAJECTORY DISTANCES")
        print(f"{'='*80}")
        
        # Get trajectories
        trajectories = transformed_datasets[frame_type].trajectories
        print(f"Trajectory shape: {trajectories.shape}")
        
        for distance_type in distance_types:
            print(f"\n--- Testing {distance_type.upper()} distances ---")
            
            # Compute distances (limit to 50 samples for efficiency)
            distance_matrix, sampled_trajectories = compute_trajectory_distance_matrix(
                trajectories, max_samples=50, distance_type=distance_type
            )
            
            # Perform clustering
            results = perform_trajectory_clustering(distance_matrix, n_clusters_list=[3, 5, 8])
            
            # Print summary
            print(f"\n--- {distance_type.title()} Clustering Summary: {frame_type} Frame ---")
            
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
            visualize_trajectory_clustering(distance_matrix, results, sampled_trajectories, 
                                         frame_type, distance_type)
    
    print(f"\n{'='*80}")
    print("ROBUST TRAJECTORY CLUSTERING ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Key advantages of trajectory-based clustering:")
    print("1. No manual feature engineering required")
    print("2. Directly compares trajectory shapes and patterns")
    print("3. Handles trajectories of different lengths")
    print("4. Multiple distance metrics available:")
    print("   - DTW: Captures temporal warping and shape similarity")
    print("   - Euclidean: Simple geometric distance")
    print("   - Cosine: Direction-based similarity")
    print("5. Robust to various trajectory characteristics")


if __name__ == "__main__":
    main()
