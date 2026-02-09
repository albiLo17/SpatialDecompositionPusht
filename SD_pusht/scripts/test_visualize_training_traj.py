#!/usr/bin/env python3
"""Test script for debugging the visualize_training_trajectory function.
This allows quick iteration without running the full training loop.

Example usage:
python SD_pusht/scripts/test_visualize_training_traj.py \
    --ckpt-path log/dp/sd-pusht-local-flow-2d-demos-50-seed0/checkpoints/best_ema_model.pt \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --sample-idx 0 \
    --out-path test_vis.png
"""

import argparse
import os
import json
import torch
import numpy as np

from SD_pusht.models import LocalFlowPolicy2D
from SD_pusht.datasets import PushTSegmentedDataset
from SD_pusht.utils.evaluation import visualize_training_trajectory


def load_config_from_checkpoint(ckpt_path):
    """
    Load config.json from the experiment directory.
    Returns None if config file doesn't exist.
    """
    # Extract experiment directory from checkpoint path
    # ckpt_path: log/dp/exp_name/checkpoints/ema_model.pt
    # config: log/dp/exp_name/config.json
    ckpt_dir = os.path.dirname(ckpt_path)  # log/dp/exp_name/checkpoints
    exp_dir = os.path.dirname(ckpt_dir)  # log/dp/exp_name
    config_path = os.path.join(exp_dir, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {config_path}")
        return config
    else:
        print(f"Config file not found at: {config_path}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Test visualize_training_trajectory function")
    
    # Required arguments
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to zarr dataset")
    
    # Checkpoint path (optional - will use fresh model if not provided)
    parser.add_argument("--ckpt-path", type=str, default=None,
                       help="Path to checkpoint file (optional, uses fresh model if not provided)")
    
    # Optional arguments with defaults
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Index of the sample to visualize")
    parser.add_argument("--out-path", type=str, default="test_vis.png",
                       help="Output path for visualization")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu), auto-detect if None")
    
    # Model hyperparameters (will try to load from config if available)
    parser.add_argument("--pred-horizon", type=int, default=16,
                       help="Prediction horizon")
    parser.add_argument("--obs-horizon", type=int, default=2,
                       help="Observation horizon")
    parser.add_argument("--action-horizon", type=int, default=8,
                       help="Action horizon")
    parser.add_argument("--obs-dim", type=int, default=5,
                       help="Observation dimension")
    parser.add_argument("--action-dim", type=int, default=2,
                       help="Action dimension")
    parser.add_argument("--fm-timesteps", type=int, default=100,
                       help="Number of timesteps for flow matching inference")
    parser.add_argument("--sigma", type=float, default=0.0,
                       help="Noise scale for flow matching paths")
    
    # LocalFlowPolicy2D specific arguments
    parser.add_argument("--use-position-decoder", action="store_true", default=True,
                       help="Use position decoder")
    parser.add_argument("--position-decoder-down-dims", type=int, nargs="+", default=[256],
                       help="Channel dimensions for position decoder U-Net")
    parser.add_argument("--position-decoder-n-groups", type=int, default=4,
                       help="Number of groups for position decoder GroupNorm")
    parser.add_argument("--position-decoder-fm-timesteps", type=int, default=8,
                       help="Number of timesteps for position decoder inference")
    parser.add_argument("--position-loss-coeff", type=float, default=1.0,
                       help="Coefficient for position loss")
    parser.add_argument("--share-noise", action="store_true",
                       help="Share noise between position and action predictions")
    parser.add_argument("--shared-noise-base", type=str, default="action",
                       choices=["action", "position", "combinatory"],
                       help="Base for noise sharing")
    parser.add_argument("--use-gt-reference-for-local-policy", action="store_true",
                       help="Use ground truth reference position for local policy conditioning")
    
    # Dataset segmentation arguments
    parser.add_argument("--contact-threshold", type=float, default=0.1,
                       help="Threshold for detecting block movement (contact)")
    parser.add_argument("--min-segment-length", type=int, default=5,
                       help="Minimum length for a valid segment")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Try to load config from checkpoint if provided
    config = None
    if args.ckpt_path:
        config = load_config_from_checkpoint(args.ckpt_path)
        
        # Override args with config values if available
        if config:
            args.pred_horizon = config.get('pred_horizon', args.pred_horizon)
            args.obs_horizon = config.get('obs_horizon', args.obs_horizon)
            args.action_horizon = config.get('action_horizon', args.action_horizon)
            args.obs_dim = config.get('obs_dim', args.obs_dim)
            args.action_dim = config.get('action_dim', args.action_dim)
            args.fm_timesteps = config.get('fm_timesteps', args.fm_timesteps)
            args.sigma = config.get('sigma', args.sigma)
            
            # LocalFlowPolicy2D specific args
            if config.get('use_position_decoder', args.use_position_decoder):
                args.use_position_decoder = True
            args.position_decoder_down_dims = config.get('position_decoder_down_dims', args.position_decoder_down_dims)
            args.position_decoder_n_groups = config.get('position_decoder_n_groups', args.position_decoder_n_groups)
            args.position_decoder_fm_timesteps = config.get('position_decoder_fm_timesteps', args.position_decoder_fm_timesteps)
            args.position_loss_coeff = config.get('position_loss_coeff', args.position_loss_coeff)
            args.share_noise = config.get('share_noise', args.share_noise)
            args.shared_noise_base = config.get('shared_noise_base', args.shared_noise_base)
            args.use_gt_reference_for_local_policy = config.get('use_gt_reference_for_local_policy', args.use_gt_reference_for_local_policy)
            
            # Dataset args
            args.contact_threshold = config.get('contact_threshold', args.contact_threshold)
            args.min_segment_length = config.get('min_segment_length', args.min_segment_length)
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Output path: {args.out_path}")
    
    # Create segmented dataset
    print("\nLoading dataset...")
    dataset = PushTSegmentedDataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_demos=None,  # Use all demos for testing
        use_contact_segmentation=True,
        contact_threshold=args.contact_threshold,
        min_segment_length=args.min_segment_length,
    )
    
    stats = getattr(dataset, "stats", None)
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    if args.sample_idx >= len(dataset):
        print(f"Error: sample_idx {args.sample_idx} is out of range (dataset has {len(dataset)} samples)")
        return
    
    # Create model
    print("\nCreating model...")
    model = LocalFlowPolicy2D(
        act_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        act_horizon=args.action_horizon,
        pred_horizon=args.pred_horizon,
        obs_dim=args.obs_dim,
        sigma=args.sigma,
        fm_timesteps=args.fm_timesteps,
        use_position_decoder=args.use_position_decoder,
        position_decoder_down_dims=args.position_decoder_down_dims,
        position_decoder_n_groups=args.position_decoder_n_groups,
        position_decoder_fm_timesteps=args.position_decoder_fm_timesteps,
        position_loss_coeff=args.position_loss_coeff,
        share_noise=args.share_noise,
        shared_noise_base=args.shared_noise_base,
        use_gt_reference_for_local_policy=args.use_gt_reference_for_local_policy,
    ).to(device)
    
    # Load checkpoint if provided
    if args.ckpt_path:
        print(f"\nLoading checkpoint from: {args.ckpt_path}")
        try:
            state_dict = torch.load(args.ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Using untrained model weights")
    else:
        print("No checkpoint provided, using untrained model weights")
    
    model.eval()
    
    # Call visualization function
    print(f"\nVisualizing sample {args.sample_idx}...")
    try:
        vis_results = visualize_training_trajectory(
            model=model,
            dataset=dataset,
            stats=stats,
            sample_idx=args.sample_idx,
            out_path=args.out_path,
            device=device,
        )
        
        print(f"\nVisualization saved to: {vis_results['image_path']}")
        print(f"GT reference position: {vis_results.get('gt_ref_pos', 'N/A')}")
        print(f"Predicted reference position: {vis_results.get('pred_ref_pos', 'N/A')}")
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

