"""Data normalization utilities for PushT datasets."""

import numpy as np


def get_data_stats(data):
    """Compute min/max statistics for normalization.
    
    Args:
        data: Array of shape (N, ...) where N is the number of samples.
        
    Returns:
        Dictionary with 'min' and 'max' keys containing per-dimension statistics.
    """
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    """Normalize data to [-1, 1] range using min/max statistics.
    
    Args:
        data: Data array to normalize.
        stats: Dictionary with 'min' and 'max' keys.
        
    Returns:
        Normalized data in [-1, 1] range.
    """
    # normalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    """Unnormalize data from [-1, 1] range back to original scale.
    
    Args:
        ndata: Normalized data in [-1, 1] range.
        stats: Dictionary with 'min' and 'max' keys.
        
    Returns:
        Unnormalized data in original scale.
    """
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

