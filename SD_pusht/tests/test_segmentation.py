#!/usr/bin/env python3
"""
Test script to verify the segmentation functionality and show statistics.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add paths for imports
REPO_ROOT = "/proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht"
SPATIALDECOMP_PATH = os.path.join(REPO_ROOT, "spatialdecomposition")
if SPATIALDECOMP_PATH not in sys.path:
    sys.path.append(SPATIALDECOMP_PATH)

from SpatialDecomposition.TwoD_table_play.data import ToyDataset

def analyze_segmented_dataset(dataset_path):
    """Analyze the segmented dataset and show statistics."""
    print(f"Loading segmented dataset: {dataset_path}")
    
    try:
        dataset = ToyDataset.from_file(dataset_path)
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Number of segments: {len(dataset)}")
        print(f"  - Trajectory shape: {dataset.trajectories.shape}")
        
        # Try to load metadata from separate file
        metadata_path = dataset_path.replace('.npz', '_metadata.npz')
        if os.path.exists(metadata_path):
            print(f"Loading metadata from: {metadata_path}")
            metadata_data = np.load(metadata_path, allow_pickle=True)
            metadata = metadata_data['segment_metadata']
            
            print(f"\nSegmentation Statistics:")
            print(f"  - Total segments: {len(metadata)}")
            
            contact_segments = sum(1 for m in metadata if m['contact_flag'])
            no_contact_segments = sum(1 for m in metadata if not m['contact_flag'])
            print(f"  - Contact segments: {contact_segments}")
            print(f"  - No-contact segments: {no_contact_segments}")
            print(f"  - Average segment length: {np.mean([m['segment_length'] for m in metadata]):.1f}")
            
            # Show first few segments
            print(f"\nFirst 10 segments:")
            for i, meta in enumerate(metadata[:10]):
                print(f"  Segment {i}: Episode {meta['original_episode']}, Contact: {meta['contact_flag']}, Length: {meta['segment_length']}")
                
            # Show contact vs no-contact distribution
            print(f"\nContact distribution by episode:")
            episode_contact = {}
            for meta in metadata:
                ep = meta['original_episode']
                if ep not in episode_contact:
                    episode_contact[ep] = {'contact': 0, 'no_contact': 0}
                if meta['contact_flag']:
                    episode_contact[ep]['contact'] += 1
                else:
                    episode_contact[ep]['no_contact'] += 1
            
            for ep in sorted(episode_contact.keys())[:5]:  # Show first 5 episodes
                counts = episode_contact[ep]
                print(f"  Episode {ep}: {counts['contact']} contact, {counts['no_contact']} no-contact segments")
                
        else:
            print("No segmentation metadata file found")
            
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")

if __name__ == "__main__":
    # Test both segmented and non-segmented datasets
    datasets = [
        "datasets/pusht_toy_dataset.npz",  # Original
        "datasets/pusht_segmented_clean.npz",  # Segmented with metadata
    ]
    
    for dataset_path in datasets:
        full_path = os.path.join(REPO_ROOT, dataset_path)
        if os.path.exists(full_path):
            print(f"\n{'='*60}")
            analyze_segmented_dataset(full_path)
        else:
            print(f"Dataset not found: {full_path}")
