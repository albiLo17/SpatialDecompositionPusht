#!/usr/bin/env python3
"""
DTW-based Trajectory Clustering Analysis

This script performs trajectory clustering using Dynamic Time Warping (DTW) distance
without requiring manual feature engineering. DTW directly compares trajectory shapes
and timing, making it ideal for trajectory analysis.

Usage:
    python dtw_clustering_analysis.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
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


class DTWTrajectoryClusterer:
    """
    DTW-based trajectory clustering analysis.
    """
    
    def __init__(self, dataset_path: str, max_samples: int = 200):
        """
        Initialize the DTW clusterer.
        
        Args:
            dataset_path: Path to the segmented dataset .npz file
            max_samples: Maximum number of samples to use (for computational efficiency)
        """
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.dataset = None
        self.transformed_datasets = {}
        self.dtw_distances = {}
        self.cluster_results = {}
        
    def load_dataset(self) -> None:
        """Load the dataset and apply transformations."""
        print("Loading dataset...")
        self.dataset = ToyDataset.from_file(self.dataset_path)
        print(f"✓ Loaded {len(self.dataset)} trajectory segments")
        
        # Apply transformations
        print("Applying coordinate frame transformations...")
        
        for reference_frame in [EndEffectorPoseFrame.FIRST, EndEffectorPoseFrame.LAST]:
            local_dataset = self.dataset.__class__(
                trajectories=self.dataset.trajectories.copy(),
                trajectory_labels_numerical=self.dataset.trajectory_labels_numerical,
                label_list=self.dataset.label_list
            )
            
            # Select reference position
            batched_local_pose = {
                EndEffectorPoseFrame.FIRST: local_dataset.trajectories[..., 0, :],
                EndEffectorPoseFrame.LAST: local_dataset.trajectories[..., -1, :],
            }[reference_frame]
            
            # Transform to local coordinates
            batched_local_pose = batched_local_pose[..., None, :]
            local_dataset.trajectories = -batched_local_pose + local_dataset.trajectories
            
            self.transformed_datasets[reference_frame.name] = local_dataset
        
        print(f"✓ Created {len(self.transformed_datasets)} transformed datasets")
        
    def compute_dtw_distances(self, frame_type: str, sample_indices: np.ndarray = None) -> np.ndarray:
        """
        Compute DTW distance matrix for trajectories.
        
        Args:
            frame_type: 'FIRST' or 'LAST' frame transformation
            sample_indices: Indices of trajectories to use (for computational efficiency)
            
        Returns:
            DTW distance matrix
        """
        if not DTW_AVAILABLE:
            raise ImportError("dtaidistance package is required for DTW clustering. Install with: pip install dtaidistance")
            
        if frame_type not in self.transformed_datasets:
            raise ValueError(f"Frame type {frame_type} not found")
            
        trajectories = self.transformed_datasets[frame_type].trajectories
        
        # Use sample indices if provided
        if sample_indices is not None:
            trajectories = trajectories[sample_indices]
            print(f"Using {len(trajectories)} sampled trajectories")
        
        print(f"Computing DTW distances for {frame_type} frame...")
        print(f"  - Trajectory shape: {trajectories.shape}")
        print(f"  - This may take a while for large datasets...")
        
        # Compute DTW distance matrix
        n_trajectories = len(trajectories)
        dtw_matrix = np.zeros((n_trajectories, n_trajectories))
        
        # Progress tracking
        total_pairs = n_trajectories * (n_trajectories - 1) // 2
        computed_pairs = 0
        
        for i in range(n_trajectories):
            for j in range(i + 1, n_trajectories):
                # Compute DTW distance between trajectories i and j
                distance = dtw.distance(trajectories[i], trajectories[j])
                dtw_matrix[i, j] = distance
                dtw_matrix[j, i] = distance  # Symmetric matrix
                
                computed_pairs += 1
                if computed_pairs % 100 == 0:
                    progress = (computed_pairs / total_pairs) * 100
                    print(f"    Progress: {computed_pairs}/{total_pairs} pairs ({progress:.1f}%)")
        
        print(f"✓ DTW distance matrix computed: {dtw_matrix.shape}")
        return dtw_matrix
        
    def perform_dtw_clustering(self, frame_type: str, dtw_matrix: np.ndarray) -> dict:
        """
        Perform clustering using DTW distance matrix.
        
        Args:
            frame_type: 'FIRST' or 'LAST' frame transformation
            dtw_matrix: Precomputed DTW distance matrix
            
        Returns:
            Dictionary containing clustering results
        """
        print(f"\nPerforming DTW-based clustering on {frame_type} frame...")
        
        results = {
            'frame_type': frame_type,
            'dtw_matrix': dtw_matrix,
            'n_samples': len(dtw_matrix)
        }
        
        # 1. K-Means with DTW distances (using precomputed distance matrix)
        print("  - Running K-Means with DTW distances...")
        kmeans_results = {}
        for n_clusters in [3, 5, 8, 10]:
            # Use AgglomerativeClustering with precomputed distances for K-Means-like behavior
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric='precomputed',
                linkage='average'
            )
            labels = agg_clustering.fit_predict(dtw_matrix)
            
            # Compute metrics
            silhouette = silhouette_score(dtw_matrix, labels, metric='precomputed')
            calinski = calinski_harabasz_score(dtw_matrix, labels)
            davies_bouldin = davies_bouldin_score(dtw_matrix, labels)
            
            kmeans_results[n_clusters] = {
                'model': agg_clustering,
                'labels': labels,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin
            }
            
        results['kmeans'] = kmeans_results
        
        # 2. HDBScan with DTW distances
        print("  - Running HDBScan with DTW distances...")
        hdbscan_results = {}
        for min_cluster_size in [5, 10, 20, 50]:
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
                calinski = calinski_harabasz_score(dtw_matrix, labels)
                davies_bouldin = davies_bouldin_score(dtw_matrix, labels)
            else:
                silhouette = calinski = davies_bouldin = 0.0
                
            hdbscan_results[min_cluster_size] = {
                'model': hdb,
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin
            }
            
        results['hdbscan'] = hdbscan_results
        
        # 3. Agglomerative Clustering with different linkage methods
        print("  - Running Agglomerative Clustering with DTW distances...")
        agg_results = {}
        for linkage in ['ward', 'complete', 'average', 'single']:
            for n_clusters in [3, 5, 8]:
                try:
                    agg_clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='precomputed',
                        linkage=linkage
                    )
                    labels = agg_clustering.fit_predict(dtw_matrix)
                    
                    silhouette = silhouette_score(dtw_matrix, labels, metric='precomputed')
                    calinski = calinski_harabasz_score(dtw_matrix, labels)
                    davies_bouldin = davies_bouldin_score(dtw_matrix, labels)
                    
                    key = f"{linkage}_{n_clusters}"
                    agg_results[key] = {
                        'model': agg_clustering,
                        'labels': labels,
                        'silhouette': silhouette,
                        'calinski_harabasz': calinski,
                        'davies_bouldin': davies_bouldin
                    }
                except Exception as e:
                    print(f"    Warning: {linkage} linkage failed: {e}")
                    
        results['agglomerative'] = agg_results
        
        # Store results
        self.cluster_results[f"{frame_type}_dtw"] = results
        
        print(f"✓ DTW clustering complete for {frame_type} frame")
        return results
        
    def visualize_dtw_clustering(self, frame_type: str, dtw_matrix: np.ndarray, 
                                results: dict, save_dir: str = "clustering_results") -> None:
        """
        Visualize DTW clustering results.
        
        Args:
            frame_type: 'FIRST' or 'LAST' frame transformation
            dtw_matrix: DTW distance matrix
            results: Clustering results
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nCreating DTW clustering visualizations for {frame_type} frame...")
        
        # Reduce DTW distances to 2D for visualization
        print("  - Computing 2D embedding of DTW distances...")
        from sklearn.manifold import MDS
        
        # Use MDS to embed DTW distances in 2D
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        dtw_2d = mds.fit_transform(dtw_matrix)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. DTW distance heatmap
        ax = axes[0]
        im = ax.imshow(dtw_matrix, cmap='viridis', aspect='auto')
        ax.set_title('DTW Distance Matrix')
        ax.set_xlabel('Trajectory Index')
        ax.set_ylabel('Trajectory Index')
        plt.colorbar(im, ax=ax)
        
        # 2. MDS embedding
        ax = axes[1]
        scatter = ax.scatter(dtw_2d[:, 0], dtw_2d[:, 1], alpha=0.6, s=20)
        ax.set_title('MDS Embedding of DTW Distances')
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        
        # 3. Best K-Means result
        kmeans_results = results['kmeans']
        best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
        kmeans_labels = kmeans_results[best_k]['labels']
        
        ax = axes[2]
        scatter = ax.scatter(dtw_2d[:, 0], dtw_2d[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6, s=20)
        ax.set_title(f'K-Means Clustering (k={best_k})\nSilhouette: {kmeans_results[best_k]["silhouette"]:.3f}')
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        plt.colorbar(scatter, ax=ax)
        
        # 4. Best HDBScan result
        hdbscan_results = results['hdbscan']
        best_min_size = max(hdbscan_results.keys(), key=lambda k: hdbscan_results[k]['silhouette'])
        hdbscan_labels = hdbscan_results[best_min_size]['labels']
        
        ax = axes[3]
        scatter = ax.scatter(dtw_2d[:, 0], dtw_2d[:, 1], c=hdbscan_labels, cmap='tab10', alpha=0.6, s=20)
        ax.set_title(f'HDBScan Clustering (min_size={best_min_size})\nSilhouette: {hdbscan_results[best_min_size]["silhouette"]:.3f}')
        ax.set_xlabel('MDS Component 1')
        ax.set_ylabel('MDS Component 2')
        plt.colorbar(scatter, ax=ax)
        
        # 5. Algorithm comparison
        ax = axes[4]
        algorithms = ['K-Means', 'HDBScan']
        silhouettes = [
            kmeans_results[best_k]['silhouette'],
            hdbscan_results[best_min_size]['silhouette']
        ]
        bars = ax.bar(algorithms, silhouettes, color=['skyblue', 'lightgreen'], alpha=0.7)
        ax.set_title('Algorithm Comparison (Silhouette Score)')
        ax.set_ylabel('Silhouette Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, silhouettes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Cluster size distribution
        ax = axes[5]
        unique_labels, counts = np.unique(kmeans_labels, return_counts=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        bars = ax.bar(range(len(unique_labels)), counts, color=colors, alpha=0.7)
        ax.set_title(f'Cluster Size Distribution (K-Means k={best_k})')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Trajectories')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom')
        
        plt.suptitle(f'DTW-based Clustering Analysis: {frame_type} Frame', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"{save_dir}/dtw_clustering_{frame_type.lower()}_frame.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        
        plt.show()
        
    def print_dtw_summary(self, frame_type: str, results: dict) -> None:
        """Print a summary of DTW clustering results."""
        print(f"\n{'='*60}")
        print(f"DTW CLUSTERING SUMMARY: {frame_type} Frame")
        print(f"{'='*60}")
        
        # K-Means results
        print(f"\n--- K-Means Results (with DTW distances) ---")
        kmeans_results = results['kmeans']
        for n_clusters in sorted(kmeans_results.keys()):
            r = kmeans_results[n_clusters]
            print(f"  k={n_clusters:2d}: Silhouette={r['silhouette']:.3f}, "
                  f"Calinski-Harabasz={r['calinski_harabasz']:.1f}, "
                  f"Davies-Bouldin={r['davies_bouldin']:.3f}")
        
        # HDBScan results
        print(f"\n--- HDBScan Results (with DTW distances) ---")
        hdbscan_results = results['hdbscan']
        for min_size in sorted(hdbscan_results.keys()):
            r = hdbscan_results[min_size]
            print(f"  min_size={min_size:3d}: {r['n_clusters']} clusters, "
                  f"{r['n_noise']} noise points, Silhouette={r['silhouette']:.3f}")
        
        # Agglomerative results
        print(f"\n--- Agglomerative Clustering Results ---")
        agg_results = results['agglomerative']
        for key in sorted(agg_results.keys()):
            r = agg_results[key]
            print(f"  {key:15s}: Silhouette={r['silhouette']:.3f}, "
                  f"Calinski-Harabasz={r['calinski_harabasz']:.1f}")
        
        # Best results
        best_kmeans = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
        best_hdbscan = max(hdbscan_results.keys(), key=lambda k: hdbscan_results[k]['silhouette'])
        best_agg = max(agg_results.keys(), key=lambda k: agg_results[k]['silhouette'])
        
        print(f"\n--- Best Results ---")
        print(f"  K-Means (k={best_kmeans}): Silhouette={kmeans_results[best_kmeans]['silhouette']:.3f}")
        print(f"  HDBScan (min_size={best_hdbscan}): Silhouette={hdbscan_results[best_hdbscan]['silhouette']:.3f}")
        print(f"  Agglomerative ({best_agg}): Silhouette={agg_results[best_agg]['silhouette']:.3f}")


def main():
    """Main function for DTW-based clustering analysis."""
    print("DTW-based Trajectory Clustering Analysis")
    print("=" * 60)
    
    if not DTW_AVAILABLE:
        print("❌ Error: dtaidistance package is required for DTW clustering.")
        print("Install it with: pip install dtaidistance")
        return
    
    # Initialize clusterer
    dataset_path = os.path.join(REPO_ROOT, "datasets", "pusht_toy_dataset_segmented.npz")
    clusterer = DTWTrajectoryClusterer(dataset_path, max_samples=200)  # Limit for computational efficiency
    
    # Load dataset
    clusterer.load_dataset()
    
    # Analyze both frame types
    for frame_type in ['FIRST', 'LAST']:
        print(f"\n{'='*80}")
        print(f"ANALYZING {frame_type} FRAME WITH DTW DISTANCES")
        print(f"{'='*80}")
        
        # Sample trajectories for computational efficiency
        n_total = len(clusterer.transformed_datasets[frame_type].trajectories)
        n_samples = min(200, n_total)  # Use up to 200 samples
        sample_indices = np.random.choice(n_total, n_samples, replace=False)
        
        print(f"Using {n_samples} sampled trajectories (from {n_total} total)")
        
        # Compute DTW distances
        dtw_matrix = clusterer.compute_dtw_distances(frame_type, sample_indices)
        
        # Perform clustering
        results = clusterer.perform_dtw_clustering(frame_type, dtw_matrix)
        
        # Print summary
        clusterer.print_dtw_summary(frame_type, results)
        
        # Create visualizations
        clusterer.visualize_dtw_clustering(frame_type, dtw_matrix, results)
    
    print(f"\n{'='*80}")
    print("DTW CLUSTERING ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Key insights:")
    print("1. DTW distances capture trajectory shape and timing similarities")
    print("2. No manual feature engineering required")
    print("3. Clustering is based on actual trajectory similarity")
    print("4. Results show which trajectories have similar movement patterns")
    print("\nDTW advantages:")
    print("- Handles trajectories of different lengths")
    print("- Captures temporal patterns and shape similarities")
    print("- Robust to time warping and speed variations")
    print("- No need for manual feature extraction")


if __name__ == "__main__":
    main()
