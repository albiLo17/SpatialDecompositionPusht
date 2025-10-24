# Trajectory Clustering Analysis

This directory contains scripts for analyzing trajectory clustering patterns in the PushT dataset using different coordinate frame transformations.

## Files

- `trajectory_clustering.py` - Comprehensive clustering analysis with multiple algorithms and feature types
- `quick_clustering_analysis.py` - Simplified analysis focused on key insights
- `requirements.txt` - Python dependencies for clustering analysis
## Quick Start

1. **Install dependencies:**
   ```bash
   cd /home/albi/Workspace/SpatialDecompositionPusht/SD_pusht/segmentation
   pip install -r requirements.txt
   ```

2. **Run quick analysis:**
   ```bash
   conda activate SD_pusht
   python quick_clustering_analysis.py
   ```

3. **Run comprehensive analysis:**
   ```bash
   python trajectory_clustering.py
   ```

## What the Analysis Does

### Coordinate Frame Transformations
- **FIRST Frame**: All trajectories start at origin (0,0)
- **LAST Frame**: All trajectories end at origin (0,0)

### Clustering Algorithms
- **K-Means**: Partition-based clustering with predefined number of clusters
- **GMM**: Gaussian Mixture Model for probabilistic clustering
- **HDBScan**: Density-based clustering that finds clusters of varying densities

### Feature Extraction
- **Simple Features**: Start/end positions, displacement, trajectory length, spatial ranges
- **Comprehensive Features**: All simple features plus velocity statistics, curvature, and PCA components

### Evaluation Metrics
- **Silhouette Score**: Measures how well-separated clusters are
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Index**: Average similarity between clusters

## Output

The analysis generates:
- **Visualizations**: PCA plots showing cluster assignments
- **Metrics**: Quantitative evaluation of clustering quality
- **Saved Plots**: PNG files in `clustering_results/` directory

## Example Results

The clustering analysis helps identify:
- **Trajectory Patterns**: Common movement patterns in the dataset
- **Coordinate Frame Effects**: How different reference frames affect clustering
- **Feature Importance**: Which trajectory characteristics are most discriminative
- **Optimal Parameters**: Best clustering parameters for each algorithm

## Customization

You can modify the analysis by:
- Changing the number of clusters/components in the clustering algorithms
- Adjusting feature extraction parameters
- Adding new clustering algorithms
- Modifying visualization styles

## Troubleshooting

If you encounter issues:
1. Make sure the dataset files exist in `/home/albi/Workspace/SpatialDecompositionPusht/datasets/`
2. Ensure you're using the correct conda environment (`SD_pusht`)
3. Check that all dependencies are installed correctly
4. Verify that the SpatialDecomposition package is properly installed
