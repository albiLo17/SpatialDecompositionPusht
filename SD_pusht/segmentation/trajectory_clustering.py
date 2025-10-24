#!/usr/bin/env python3
"""
Trajectory Clustering Analysis

This script loads the PushT segmented dataset, applies coordinate frame transformations,
and performs clustering analysis using different algorithms (K-Means, GMM, HDBScan).
It visualizes the clustering results and provides insights into trajectory patterns.

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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


class TrajectoryClusterer:
    """
    A comprehensive trajectory clustering analysis class.
    """
    
    def __init__(self, dataset_path: str, metadata_path: Optional[str] = None):
        """
        Initialize the trajectory clusterer.
        
        Args:
            dataset_path: Path to the segmented dataset .npz file
            metadata_path: Path to the metadata file (optional)
        """
        self.dataset_path = dataset_path
        self.metadata_path = metadata_path
        self.dataset = None
        self.metadata = None
        self.transformed_datasets = {}
        self.features = {}
        self.cluster_results = {}
        self.scalers = {}
        
    def load_dataset(self) -> None:
        """Load the dataset and metadata."""
        print("Loading dataset...")
        self.dataset = ToyDataset.from_file(self.dataset_path)
        print(f"✓ Loaded dataset with {len(self.dataset)} segments")
        print(f"  - Trajectory shape: {self.dataset.trajectories.shape}")
        
        if self.metadata_path and os.path.exists(self.metadata_path):
            self.metadata = np.load(self.metadata_path, allow_pickle=True)
            print(f"✓ Loaded metadata with {len(self.metadata['segment_metadata'])} segments")
        else:
            print("⚠ No metadata file found")
            
    def apply_transformations(self) -> None:
        """Apply coordinate frame transformations."""
        print("\nApplying coordinate frame transformations...")
        
        # Define reference frames
        reference_frames = [EndEffectorPoseFrame.FIRST, EndEffectorPoseFrame.LAST]
        
        for reference_frame in reference_frames:
            # Deep copy the dataset
            local_dataset = self.dataset.__class__(
                trajectories=self.dataset.trajectories.copy(),
                trajectory_labels_numerical=self.dataset.trajectory_labels_numerical,
                label_list=self.dataset.label_list
            )
            
            # Select reference position based on frame type
            batched_local_pose = {
                EndEffectorPoseFrame.FIRST: local_dataset.trajectories[..., 0, :],  # First frame
                EndEffectorPoseFrame.LAST: local_dataset.trajectories[..., -1, :],   # Last frame
            }[reference_frame]
            
            # Add back a time dimension for broadcasting: [N, D] -> [N, 1, D]
            batched_local_pose = batched_local_pose[..., None, :]
            
            # Transform trajectories to local coordinates
            local_dataset.trajectories = -batched_local_pose + local_dataset.trajectories
            
            # Store with reference frame name as key
            self.transformed_datasets[reference_frame.name] = local_dataset
            
        print(f"✓ Created {len(self.transformed_datasets)} transformed datasets")
        print(f"  - Reference frames: {list(self.transformed_datasets.keys())}")
        
    def extract_features(self, frame_type: str, feature_type: str = "comprehensive") -> np.ndarray:
        """
        Extract features from trajectories for clustering.
        
        Args:
            frame_type: 'FIRST' or 'LAST' frame transformation
            feature_type: Type of features to extract
            
        Returns:
            Feature matrix [N, D] where N is number of trajectories, D is feature dimension
        """
        if frame_type not in self.transformed_datasets:
            raise ValueError(f"Frame type {frame_type} not found in transformed datasets")
            
        dataset = self.transformed_datasets[frame_type]
        trajectories = dataset.trajectories  # [N, T, 2]
        
        if feature_type == "comprehensive":
            # Comprehensive feature set
            features = []
            
            # 1. Start and end positions
            start_pos = trajectories[:, 0, :]  # [N, 2]
            end_pos = trajectories[:, -1, :]    # [N, 2]
            features.extend([start_pos[:, 0], start_pos[:, 1], end_pos[:, 0], end_pos[:, 1]])
            
            # 2. Displacement vector
            displacement = end_pos - start_pos  # [N, 2]
            displacement_magnitude = np.linalg.norm(displacement, axis=1)  # [N]
            displacement_angle = np.arctan2(displacement[:, 1], displacement[:, 0])  # [N]
            features.extend([displacement_magnitude, displacement_angle])
            
            # 3. Trajectory statistics
            traj_lengths = np.linalg.norm(np.diff(trajectories, axis=1), axis=2).sum(axis=1)  # [N]
            traj_ranges_x = np.max(trajectories[:, :, 0], axis=1) - np.min(trajectories[:, :, 0], axis=1)  # [N]
            traj_ranges_y = np.max(trajectories[:, :, 1], axis=1) - np.min(trajectories[:, :, 1], axis=1)  # [N]
            features.extend([traj_lengths, traj_ranges_x, traj_ranges_y])
            
            # 4. Velocity statistics
            velocities = np.diff(trajectories, axis=1)  # [N, T-1, 2]
            vel_magnitudes = np.linalg.norm(velocities, axis=2)  # [N, T-1]
            mean_vel = np.mean(vel_magnitudes, axis=1)  # [N]
            max_vel = np.max(vel_magnitudes, axis=1)  # [N]
            vel_std = np.std(vel_magnitudes, axis=1)  # [N]
            features.extend([mean_vel, max_vel, vel_std])
            
            # 5. Curvature features
            # Compute curvature at each point
            curvatures = []
            for i in range(len(trajectories)):
                traj = trajectories[i]  # [T, 2]
                if len(traj) > 2:
                    # Compute first and second derivatives
                    first_deriv = np.gradient(traj, axis=0)
                    second_deriv = np.gradient(first_deriv, axis=0)
                    
                    # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
                    numerator = np.abs(first_deriv[:, 0] * second_deriv[:, 1] - first_deriv[:, 1] * second_deriv[:, 0])
                    denominator = (first_deriv[:, 0]**2 + first_deriv[:, 1]**2)**(3/2)
                    denominator = np.where(denominator == 0, 1e-8, denominator)  # Avoid division by zero
                    curvature = numerator / denominator
                    curvatures.append(np.mean(curvature))
                else:
                    curvatures.append(0.0)
            
            features.append(np.array(curvatures))
            
            # 6. PCA features (first few principal components)
            traj_flat = trajectories.reshape(len(trajectories), -1)  # [N, T*2]
            pca = PCA(n_components=min(5, traj_flat.shape[1]))
            pca_features = pca.fit_transform(traj_flat)
            for i in range(pca_features.shape[1]):
                features.append(pca_features[:, i])
            
            # Stack all features
            feature_matrix = np.column_stack(features)
            
        elif feature_type == "simple":
            # Simple feature set: just start, end, and displacement
            start_pos = trajectories[:, 0, :]
            end_pos = trajectories[:, -1, :]
            displacement = end_pos - start_pos
            displacement_magnitude = np.linalg.norm(displacement, axis=1)
            displacement_angle = np.arctan2(displacement[:, 1], displacement[:, 0])
            
            feature_matrix = np.column_stack([
                start_pos[:, 0], start_pos[:, 1],  # start x, y
                end_pos[:, 0], end_pos[:, 1],      # end x, y
                displacement_magnitude,             # displacement magnitude
                displacement_angle                  # displacement angle
            ])
            
        elif feature_type == "raw":
            # Raw trajectory points (flattened)
            feature_matrix = trajectories.reshape(len(trajectories), -1)
            
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        return feature_matrix
        
    def perform_clustering(self, frame_type: str, feature_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform clustering analysis on the specified frame type.
        
        Args:
            frame_type: 'FIRST' or 'LAST' frame transformation
            feature_type: Type of features to extract
            
        Returns:
            Dictionary containing clustering results
        """
        print(f"\nPerforming clustering on {frame_type} frame with {feature_type} features...")
        
        # Extract features
        features = self.extract_features(frame_type, feature_type)
        print(f"  - Feature matrix shape: {features.shape}")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers[f"{frame_type}_{feature_type}"] = scaler
        
        # Store features
        self.features[f"{frame_type}_{feature_type}"] = features_scaled
        
        results = {
            'frame_type': frame_type,
            'feature_type': feature_type,
            'features': features_scaled,
            'n_samples': len(features_scaled),
            'n_features': features_scaled.shape[1]
        }
        
        # 1. K-Means clustering
        print("  - Running K-Means...")
        kmeans_results = {}
        for n_clusters in [3, 5, 8, 10]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            # Compute metrics
            silhouette = silhouette_score(features_scaled, labels)
            calinski = calinski_harabasz_score(features_scaled, labels)
            davies_bouldin = davies_bouldin_score(features_scaled, labels)
            
            kmeans_results[n_clusters] = {
                'model': kmeans,
                'labels': labels,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin,
                'inertia': kmeans.inertia_
            }
            
        results['kmeans'] = kmeans_results
        
        # 2. Gaussian Mixture Model
        print("  - Running Gaussian Mixture Model...")
        gmm_results = {}
        for n_components in [3, 5,]: #, 8, 10]:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            labels = gmm.fit_predict(features_scaled)
            
            # Compute metrics
            silhouette = silhouette_score(features_scaled, labels)
            calinski = calinski_harabasz_score(features_scaled, labels)
            davies_bouldin = davies_bouldin_score(features_scaled, labels)
            
            gmm_results[n_components] = {
                'model': gmm,
                'labels': labels,
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin,
                'aic': gmm.aic(features_scaled),
                'bic': gmm.bic(features_scaled)
            }
            
        results['gmm'] = gmm_results
        
        # 3. HDBScan
        print("  - Running HDBScan...")
        hdbscan_results = {}
        for min_cluster_size in [10, 20, 50, 100]:
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
            labels = hdb.fit_predict(features_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                # Compute metrics (only if we have clusters)
                silhouette = silhouette_score(features_scaled, labels)
                calinski = calinski_harabasz_score(features_scaled, labels)
                davies_bouldin = davies_bouldin_score(features_scaled, labels)
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
        
        # Store results
        self.cluster_results[f"{frame_type}_{feature_type}"] = results
        
        print(f"✓ Clustering complete for {frame_type} frame")
        return results
        
    def visualize_clusters(self, frame_type: str, feature_type: str = "comprehensive", 
                          save_plots: bool = True, output_dir: str = "clustering_results") -> None:
        """
        Visualize clustering results.
        
        Args:
            frame_type: 'FIRST' or 'LAST' frame transformation
            feature_type: Type of features used
            save_plots: Whether to save plots to disk
            output_dir: Directory to save plots
        """
        if f"{frame_type}_{feature_type}" not in self.cluster_results:
            print(f"No clustering results found for {frame_type}_{feature_type}")
            return
            
        results = self.cluster_results[f"{frame_type}_{feature_type}"]
        features = results['features']
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            
        # 1. PCA visualization
        print(f"\nCreating visualizations for {frame_type} frame...")
        
        # Reduce to 2D for visualization
        pca_2d = PCA(n_components=2)
        features_2d = pca_2d.fit_transform(features)
        
        # Create subplots for different clustering methods
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # K-Means visualization
        kmeans_results = results['kmeans']
        best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
        kmeans_labels = kmeans_results[best_k]['labels']
        
        ax = axes[0]
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6)
        ax.set_title(f'K-Means (k={best_k})\nSilhouette: {kmeans_results[best_k]["silhouette"]:.3f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax)
        
        # GMM visualization
        gmm_results = results['gmm']
        best_components = max(gmm_results.keys(), key=lambda k: gmm_results[k]['silhouette'])
        gmm_labels = gmm_results[best_components]['labels']
        
        ax = axes[1]
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=gmm_labels, cmap='tab10', alpha=0.6)
        ax.set_title(f'GMM (components={best_components})\nSilhouette: {gmm_results[best_components]["silhouette"]:.3f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax)
        
        # HDBScan visualization
        hdbscan_results = results['hdbscan']
        best_min_size = max(hdbscan_results.keys(), key=lambda k: hdbscan_results[k]['silhouette'])
        hdbscan_labels = hdbscan_results[best_min_size]['labels']
        
        ax = axes[2]
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=hdbscan_labels, cmap='tab10', alpha=0.6)
        ax.set_title(f'HDBScan (min_size={best_min_size})\nSilhouette: {hdbscan_results[best_min_size]["silhouette"]:.3f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter, ax=ax)
        
        # Metrics comparison
        ax = axes[3]
        methods = ['K-Means', 'GMM', 'HDBScan']
        silhouettes = [
            kmeans_results[best_k]['silhouette'],
            gmm_results[best_components]['silhouette'],
            hdbscan_results[best_min_size]['silhouette']
        ]
        bars = ax.bar(methods, silhouettes, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Silhouette Score Comparison')
        ax.set_ylabel('Silhouette Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, silhouettes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Cluster size distribution
        ax = axes[4]
        kmeans_counts = np.bincount(kmeans_labels)
        ax.bar(range(len(kmeans_counts)), kmeans_counts, color='skyblue', alpha=0.7)
        ax.set_title(f'K-Means Cluster Sizes (k={best_k})')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Trajectories')
        
        # Feature importance (if using comprehensive features)
        if feature_type == "comprehensive":
            ax = axes[5]
            # Get feature names (simplified)
            feature_names = [
                'Start X', 'Start Y', 'End X', 'End Y',
                'Displacement Mag', 'Displacement Ang',
                'Traj Length', 'Range X', 'Range Y',
                'Mean Vel', 'Max Vel', 'Vel Std', 'Curvature'
            ] + [f'PC{i+1}' for i in range(5)]
            
            # Use PCA to get feature importance
            pca_full = PCA()
            pca_full.fit(features)
            importance = np.abs(pca_full.components_[0])  # First principal component
            importance = importance[:len(feature_names)]  # Truncate to match feature names
            
            ax.barh(range(len(importance)), importance, color='orange', alpha=0.7)
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names)
            ax.set_title('Feature Importance (PC1)')
            ax.set_xlabel('Absolute Loading')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Feature importance\nnot available for\nsimple features', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle(f'Clustering Analysis: {frame_type} Frame ({feature_type} features)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            filename = f"{output_dir}/clustering_{frame_type}_{feature_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  - Saved plot: {filename}")
        
        plt.show()
        
    def print_clustering_summary(self, frame_type: str, feature_type: str = "comprehensive") -> None:
        """Print a summary of clustering results."""
        if f"{frame_type}_{feature_type}" not in self.cluster_results:
            print(f"No clustering results found for {frame_type}_{feature_type}")
            return
            
        results = self.cluster_results[f"{frame_type}_{feature_type}"]
        
        print(f"\n{'='*60}")
        print(f"CLUSTERING SUMMARY: {frame_type} Frame ({feature_type} features)")
        print(f"{'='*60}")
        print(f"Dataset: {results['n_samples']} trajectories, {results['n_features']} features")
        
        # K-Means results
        print(f"\n--- K-Means Results ---")
        kmeans_results = results['kmeans']
        for n_clusters in sorted(kmeans_results.keys()):
            r = kmeans_results[n_clusters]
            print(f"  k={n_clusters:2d}: Silhouette={r['silhouette']:.3f}, "
                  f"Calinski-Harabasz={r['calinski_harabasz']:.1f}, "
                  f"Davies-Bouldin={r['davies_bouldin']:.3f}")
        
        # GMM results
        print(f"\n--- GMM Results ---")
        gmm_results = results['gmm']
        for n_components in sorted(gmm_results.keys()):
            r = gmm_results[n_components]
            print(f"  components={n_components:2d}: Silhouette={r['silhouette']:.3f}, "
                  f"AIC={r['aic']:.1f}, BIC={r['bic']:.1f}")
        
        # HDBScan results
        print(f"\n--- HDBScan Results ---")
        hdbscan_results = results['hdbscan']
        for min_size in sorted(hdbscan_results.keys()):
            r = hdbscan_results[min_size]
            print(f"  min_size={min_size:3d}: {r['n_clusters']} clusters, "
                  f"{r['n_noise']} noise points, Silhouette={r['silhouette']:.3f}")
        
        # Best results
        best_kmeans = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
        best_gmm = max(gmm_results.keys(), key=lambda k: gmm_results[k]['silhouette'])
        best_hdbscan = max(hdbscan_results.keys(), key=lambda k: hdbscan_results[k]['silhouette'])
        
        print(f"\n--- Best Results ---")
        print(f"  K-Means (k={best_kmeans}): Silhouette={kmeans_results[best_kmeans]['silhouette']:.3f}")
        print(f"  GMM (components={best_gmm}): Silhouette={gmm_results[best_gmm]['silhouette']:.3f}")
        print(f"  HDBScan (min_size={best_hdbscan}): Silhouette={hdbscan_results[best_hdbscan]['silhouette']:.3f}")


def main():
    """Main function to run the clustering analysis."""
    print("Trajectory Clustering Analysis")
    print("=" * 50)
    
    # Initialize clusterer
    dataset_path = os.path.join(REPO_ROOT, "datasets", "pusht_toy_dataset_segmented.npz")
    metadata_path = os.path.join(REPO_ROOT, "datasets", "pusht_toy_dataset_segmented_metadata.npz")
    
    clusterer = TrajectoryClusterer(dataset_path, metadata_path)
    
    # Load dataset
    clusterer.load_dataset()
    
    # Apply transformations
    clusterer.apply_transformations()
    
    # Perform clustering on both frame types
    frame_types = ['FIRST', 'LAST']
    feature_types = ['comprehensive', 'simple']
    
    for frame_type in frame_types:
        for feature_type in feature_types:
            print(f"\n{'='*80}")
            print(f"ANALYZING: {frame_type} Frame with {feature_type} features")
            print(f"{'='*80}")
            
            # Perform clustering
            results = clusterer.perform_clustering(frame_type, feature_type)
            
            # Print summary
            clusterer.print_clustering_summary(frame_type, feature_type)
            
            # Create visualizations
            clusterer.visualize_clusters(frame_type, feature_type, save_plots=True)
    
    print(f"\n{'='*80}")
    print("CLUSTERING ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print("Results saved in 'clustering_results/' directory")
    print("Check the generated plots for visual insights into trajectory clusters.")


if __name__ == "__main__":
    main()
