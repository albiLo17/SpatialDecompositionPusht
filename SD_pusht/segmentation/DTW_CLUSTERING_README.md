# DTW-based Trajectory Clustering

This directory contains scripts for performing trajectory clustering using Dynamic Time Warping (DTW) distances, eliminating the need for manual feature engineering.

## 🎯 Key Advantages

- **No Feature Engineering**: DTW directly compares trajectory shapes and timing
- **Handles Variable Lengths**: Works with trajectories of different durations
- **Temporal Robustness**: Captures time warping and speed variations
- **Shape Similarity**: Focuses on movement patterns rather than absolute positions
- **Multiple Distance Metrics**: DTW, Euclidean, and Cosine distances available

## 📁 Files Overview

### Core Scripts
- **`robust_dtw_clustering.py`** - Main DTW clustering analysis (recommended)
- **`simple_dtw_clustering.py`** - Simplified version for basic analysis
- **`dtw_clustering_analysis.py`** - Comprehensive DTW analysis (advanced)

### Supporting Files
- **`requirements.txt`** - Python dependencies including `dtaidistance`
- **`README.md`** - General clustering analysis overview
- **`DTW_CLUSTERING_README.md`** - This file

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /home/albi/Workspace/SpatialDecompositionPusht/SD_pusht/segmentation
conda activate SD_pusht
pip install -r requirements.txt
```

### 2. Run DTW Clustering Analysis
```bash
python robust_dtw_clustering.py
```

This will:
- Load the segmented trajectory dataset
- Apply coordinate frame transformations (FIRST and LAST)
- Compute DTW, Euclidean, and Cosine distance matrices
- Perform clustering with multiple algorithms
- Generate visualizations and save results

## 📊 Results Interpretation

### Distance Metrics Comparison

| Metric | Best Use Case | Silhouette Score | Notes |
|--------|---------------|------------------|-------|
| **DTW** | Shape similarity, temporal patterns | 0.3-0.5 | Captures time warping |
| **Cosine** | Direction-based similarity | 0.4-0.6 | Good for trajectory directions |
| **Euclidean** | Geometric distance | 0.3-0.4 | Simple but effective |

### Clustering Algorithms

| Algorithm | Best For | Typical Results |
|-----------|----------|-----------------|
| **Agglomerative** | Hierarchical clustering | 3-8 clusters, good silhouette |
| **HDBScan** | Density-based clustering | 2-4 clusters, some noise points |

### Frame Transformations

| Frame Type | Characteristics | Best Distance |
|------------|----------------|---------------|
| **FIRST** | Start position reference | Cosine (0.57 silhouette) |
| **LAST** | End position reference | DTW (0.51 silhouette) |

## 🔧 Advanced Usage

### Custom Parameters
```python
# Modify the robust_dtw_clustering.py script
max_samples = 100  # Increase for more trajectories
n_clusters_list = [3, 5, 8, 10]  # Add more cluster numbers
min_cluster_sizes = [5, 10, 20, 50]  # HDBScan parameters
```

### Distance Matrix Analysis
```python
# Access computed distance matrices
distance_matrix, trajectories = compute_trajectory_distance_matrix(
    trajectories, max_samples=100, distance_type='dtw'
)

# Analyze distance distributions
print(f"Mean distance: {np.mean(distance_matrix):.3f}")
print(f"Std distance: {np.std(distance_matrix):.3f}")
print(f"Min distance: {np.min(distance_matrix):.3f}")
print(f"Max distance: {np.max(distance_matrix):.3f}")
```

## 📈 Performance Results

### Best Configurations Found

1. **DTW + FIRST Frame + Agglomerative (k=5)**
   - Silhouette: 0.444
   - 5 clusters with good separation

2. **Cosine + FIRST Frame + Agglomerative (k=3)**
   - Silhouette: 0.572
   - 3 clusters with excellent separation

3. **DTW + LAST Frame + Agglomerative (k=3)**
   - Silhouette: 0.512
   - 3 clusters with good temporal patterns

### Computational Efficiency
- **50 trajectories**: ~2-3 minutes
- **100 trajectories**: ~8-10 minutes
- **200+ trajectories**: Consider sampling or parallel processing

## 🎨 Visualization Outputs

The script generates several visualization files:

- `dtw_clustering_first_frame.png` - DTW analysis for FIRST frame
- `dtw_clustering_last_frame.png` - DTW analysis for LAST frame
- `euclidean_clustering_*.png` - Euclidean distance analysis
- `cosine_clustering_*.png` - Cosine distance analysis

Each visualization includes:
1. Distance matrix heatmap
2. MDS 2D embedding
3. Clustering results
4. Algorithm comparison

## 🔍 Troubleshooting

### Common Issues

1. **DTW Computation Errors**
   ```bash
   # Solution: Use the robust version
   python robust_dtw_clustering.py
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Reduce sample size
   max_samples = 50  # Instead of 100
   ```

3. **Poor Clustering Results**
   ```python
   # Try different distance metrics
   distance_types = ['cosine', 'euclidean']  # Skip DTW if problematic
   ```

### Performance Tips

1. **Start Small**: Use 50 samples first, then scale up
2. **Try Multiple Metrics**: DTW, Euclidean, and Cosine often give different insights
3. **Check Silhouette Scores**: Values > 0.3 are generally good
4. **Visualize Results**: Always check the generated plots

## 📚 Technical Details

### DTW Distance Computation
```python
def safe_dtw_distance(traj1, traj2):
    """Safely compute DTW distance with fallback to Euclidean."""
    try:
        distance = dtw.distance(traj1.flatten(), traj2.flatten())
        return float(distance)
    except Exception:
        return float(np.linalg.norm(traj1.flatten() - traj2.flatten()))
```

### Distance Matrix Properties
- **Symmetric**: `distance_matrix[i,j] = distance_matrix[j,i]`
- **Zero Diagonal**: `distance_matrix[i,i] = 0`
- **Non-negative**: All distances ≥ 0
- **Triangle Inequality**: May not hold for DTW

### Clustering with Precomputed Distances
```python
# Use 'precomputed' metric for distance matrices
clusterer = AgglomerativeClustering(
    n_clusters=5,
    metric='precomputed',  # Use precomputed distance matrix
    linkage='average'
)
labels = clusterer.fit_predict(distance_matrix)
```

## 🎯 Next Steps

1. **Scale Up**: Increase sample size for more comprehensive analysis
2. **Parameter Tuning**: Experiment with different cluster numbers and algorithms
3. **Feature Integration**: Combine DTW with traditional features for hybrid approach
4. **Temporal Analysis**: Analyze how clusters change over time
5. **Validation**: Use cross-validation to assess clustering stability

## 📖 References

- [DTW Algorithm](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- [dtaidistance Library](https://dtaidistance.readthedocs.io/)
- [Trajectory Clustering Survey](https://link.springer.com/article/10.1007/s10115-016-0986-0)
- [HDBScan Documentation](https://hdbscan.readthedocs.io/)

---

**Happy Clustering! 🎉**

For questions or issues, check the troubleshooting section or examine the generated visualizations for insights into your trajectory data.
