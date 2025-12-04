import sys
import os
import numpy as np

# Add paths
REPO_ROOT = "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht"
sys.path.append(os.path.join(REPO_ROOT, "spatialdecomposition"))

from SpatialDecomposition.TwoD_table_play.data import ToyDataset

# Load dataset and metadata
segmented_path = os.path.join(REPO_ROOT, "datasets", "pusht_toy_dataset_segmented.npz")
metadata_path = os.path.join(REPO_ROOT, "datasets", "pusht_toy_dataset_segmented_metadata.npz")

dataset = ToyDataset.from_file(segmented_path)
metadata_data = np.load(metadata_path, allow_pickle=True)
segment_metadata = metadata_data['segment_metadata']

print(f"Dataset length: {len(dataset)}")
print(f"Metadata length: {len(segment_metadata)}")

# Check first metadata entry
meta = segment_metadata[0]
print(f"\nFirst metadata entry type: {type(meta)}")
print(f"First metadata: {meta}")

# Test visualization code
print("\n=== Testing visualization ===")
sample = dataset[0]
trajectory = sample.trajectory

print(f"Trajectory shape: {trajectory.shape}")
print(f"Trajectory type: {type(trajectory)}")
print(f"First point: ({trajectory[0, 0]}, {trajectory[0, 1]})")
print(f"Contact flag: {meta['contact_flag']}")
print(f"Episode: {meta['original_episode']}")
print(f"Segment idx: {meta['segment_idx']}")

print("\n✓ All metadata fields accessible!")

