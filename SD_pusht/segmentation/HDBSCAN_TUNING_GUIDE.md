# HDBSCAN Hyperparameter Tuning Guide

This guide explains how to properly tune HDBSCAN hyperparameters for trajectory clustering.

## Key Hyperparameters

### 1. `min_cluster_size` (Most Critical)
**What it does**: Minimum number of points required to form a cluster
**Impact**: 
- Lower values → more, smaller clusters
- Higher values → fewer, larger clusters
- Too low → noisy clusters
- Too high → important patterns missed

**Tuning Strategy**:
- Start with 10-20% of your dataset size
- For 1291 trajectories: try 10, 20, 50, 100, 200
- Adjust based on domain knowledge of expected cluster sizes

**Example**:
```python
# For 1291 trajectories
min_cluster_sizes = [10, 20, 50, 100, 200]  # 0.8%, 1.5%, 3.9%, 7.7%, 15.5% of data
```

### 2. `min_samples` (Second Most Critical)
**What it does**: Number of samples in a neighborhood for a point to be considered a core point
**Impact**:
- Controls cluster density requirements
- Lower values → more sensitive to noise
- Higher values → more selective clustering

**Tuning Strategy**:
- Usually set to 1/3 to 1/2 of `min_cluster_size`
- Must be < `min_cluster_size`
- Try: 3, 5, 10, 15, 20

**Example**:
```python
# If min_cluster_size = 20, try min_samples = 5, 10, 15
min_samples_list = [3, 5, 10, 15, 20]
```

### 3. `cluster_selection_epsilon`
**What it does**: Distance threshold for cluster selection
**Impact**:
- 0.0 → automatic selection (recommended)
- Higher values → more cluster merging
- Lower values → more separate clusters

**Tuning Strategy**:
- Start with 0.0 (automatic)
- If too many small clusters, try 0.1-0.3
- If too few clusters, try 0.0 or negative values

**Example**:
```python
cluster_selection_epsilons = [0.0, 0.1, 0.2, 0.3, 0.5]
```

### 4. `metric`
**What it does**: Distance metric for clustering
**Options**:
- `'euclidean'` → Standard Euclidean distance (recommended)
- `'manhattan'` → L1 distance
- `'cosine'` → Cosine similarity (good for high-dimensional data)
- `'precomputed'` → Use precomputed distance matrix

**Tuning Strategy**:
- Use `'euclidean'` for most trajectory data
- Use `'cosine'` if features are high-dimensional or normalized
- Use `'manhattan'` if you want L1 distance

**Example**:
```python
metrics = ['euclidean', 'manhattan']  # Note: 'cosine' may not be supported in all HDBSCAN versions
```

### 5. `cluster_selection_method`
**What it does**: Method for selecting clusters from the condensed tree
**Options**:
- `'eom'` → Excess of Mass (default, recommended)
- `'leaf'` → Leaf selection

**Tuning Strategy**:
- Use `'eom'` (default) for most cases
- Use `'leaf'` if you want more conservative clustering

## Tuning Process

### Step 1: Start with Default Values
```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,        # 1.5% of 1291 trajectories
    min_samples=5,              # 1/4 of min_cluster_size
    cluster_selection_epsilon=0.0,  # Automatic
    metric='euclidean'         # Standard metric
)
```

### Step 2: Tune `min_cluster_size`
- Test values: [5, 10, 20, 50, 100, 200]
- Look for good silhouette scores
- Balance between cluster count and quality

### Step 3: Tune `min_samples`
- Test values: [3, 5, 10, 15, 20]
- Must be < `min_cluster_size`
- Look for stable clustering results

### Step 4: Tune `cluster_selection_epsilon`
- Test values: [0.0, 0.1, 0.2, 0.3, 0.5]
- Adjust based on cluster merging needs

### Step 5: Tune `metric`
- Test: ['euclidean', 'manhattan', 'cosine']
- Choose based on data characteristics

## Evaluation Metrics

### Primary Metrics
1. **Silhouette Score**: Measures cluster separation (-1 to 1, higher is better)
2. **Number of Clusters**: Should be meaningful for your domain
3. **Noise Percentage**: Should be reasonable (10-30% is typical)

### Secondary Metrics
1. **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
2. **Davies-Bouldin Index**: Average similarity between clusters (lower is better)

## Example Tuning Results

For trajectory clustering with 1291 segments:

### Good Parameters
```python
# Best result from tuning
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,        # 1.5% of data
    min_samples=5,              # 1/4 of min_cluster_size
    cluster_selection_epsilon=0.0,  # Automatic selection
    metric='euclidean'          # Standard metric
)
# Results: 7 clusters, 0.023 silhouette, 15.2% noise
```

### Parameter Effects
- **min_cluster_size=10**: More clusters, higher noise
- **min_cluster_size=50**: Fewer clusters, lower noise
- **min_samples=3**: More sensitive to noise
- **min_samples=15**: More selective clustering
- **epsilon=0.1**: More cluster merging
- **epsilon=0.0**: Automatic selection

## Best Practices

### 1. Start Conservative
- Begin with larger `min_cluster_size` values
- Use automatic `cluster_selection_epsilon=0.0`
- Use standard `metric='euclidean'`

### 2. Iterative Tuning
- Fix one parameter at a time
- Use grid search for systematic exploration
- Validate results with domain knowledge

### 3. Consider Data Characteristics
- **High-dimensional data**: Try `metric='cosine'`
- **Noisy data**: Increase `min_cluster_size` and `min_samples`
- **Sparse data**: Decrease `min_cluster_size`

### 4. Validation
- Check cluster interpretability
- Verify cluster stability
- Compare with other algorithms (K-Means, GMM)

## Common Issues and Solutions

### Issue: Too Many Small Clusters
**Solution**: Increase `min_cluster_size` or `cluster_selection_epsilon`

### Issue: Too Few Clusters
**Solution**: Decrease `min_cluster_size` or set `cluster_selection_epsilon=0.0`

### Issue: High Noise Percentage
**Solution**: Increase `min_samples` or `min_cluster_size`

### Issue: Poor Cluster Quality
**Solution**: Try different `metric` or feature preprocessing

## Running the Tuning Script

```bash
cd /home/albi/Workspace/SpatialDecompositionPusht/SD_pusht/segmentation
conda activate SD_pusht
python hdbscan_tuning.py
```

This will:
1. Test all parameter combinations
2. Show results for each combination
3. Identify the best parameters
4. Create visualizations of parameter effects
5. Provide recommendations

## Expected Results

For trajectory clustering, you should expect:
- **5-15 clusters** (depending on data complexity)
- **10-30% noise** (reasonable for real-world data)
- **Silhouette score 0.0-0.3** (good for density-based clustering)
- **Stable results** across similar parameter values

The key is finding parameters that produce meaningful, interpretable clusters for your specific use case.
