#!/usr/bin/env python3
"""Evaluation script for local flow 2D policy on PushT."""
"""
Example usage:
python SD_pusht/scripts/eval_local_flow_2d.py \
    --ckpt-path log/dp/sd-pusht-local-flow-2d_V3-demos-100-seed1234-no-share-noise-pred-ref/checkpoints/best_ema_model.pt \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --num-envs 32 \
    --max-steps 300
"""

import argparse
import os
import json
import torch
import numpy as np
from datetime import datetime

from SD_pusht.models import LocalFlowPolicy2D
from SD_pusht.datasets import PushTSegmentedDatasetSimple
from SD_pusht.utils.evaluation import evaluate_local_flow_2d


def load_config_from_checkpoint(ckpt_path):
    """
    Load config.json from the experiment directory.
    Returns None if config file doesn't exist.
    """
    # Extract experiment directory from checkpoint path
    # ckpt_path: log/dp/exp_name/checkpoints/best_ema_model.pt
    # config: log/dp/exp_name/config.json
    # Note: exp_name now includes aggregation method (median/knn) in the name
    # Format: sd-pusht-local-flow-2d_V3-demos-{N}-seed{S}-{noise}-{ref}-{film}-{agg}
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


if __name__ == "__main__":
    # First parse args minimally to get checkpoint path
    parser = argparse.ArgumentParser(description="Evaluate local flow 2D policy on PushT")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to checkpoint file")
    initial_args, remaining_args = parser.parse_known_args()
    
    # Try to load config from experiment directory
    config = load_config_from_checkpoint(initial_args.ckpt_path)
    
    if config:
        print("\n" + "="*50)
        print("Loaded Config from Training (for verification):")
        print("="*50)
        print(f"  position_decoder_num_particles: {config.get('position_decoder_num_particles', 'NOT FOUND')}")
        print(f"  position_decoder_particles_aggregation: {config.get('position_decoder_particles_aggregation', 'NOT FOUND')}")
        print(f"  use_position_decoder: {config.get('use_position_decoder', 'NOT FOUND')}")
        print(f"  share_noise: {config.get('share_noise', 'NOT FOUND')}")
        print(f"  use_gt_reference_for_local_policy: {config.get('use_gt_reference_for_local_policy', 'NOT FOUND')}")
        print(f"  use_film_conditioning: {config.get('use_film_conditioning', 'NOT FOUND')}")
        print(f"  fm_timesteps: {config.get('fm_timesteps', 'NOT FOUND')}")
        print(f"  position_decoder_fm_timesteps: {config.get('position_decoder_fm_timesteps', 'NOT FOUND')}")
        print(f"  disable_reference_conditioning: {config.get('disable_reference_conditioning', 'NOT FOUND (defaults to False for old models)')}")
        print("="*50 + "\n")
    
    # Now parse full args, using config as defaults if available
    parser = argparse.ArgumentParser(description="Evaluate local flow 2D policy on PushT")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--dataset-path", type=str, 
                       default=config.get('dataset_path', "datasets/pusht_cchi_v7_replay.zarr.zip") if config else "datasets/pusht_cchi_v7_replay.zarr.zip",
                       help="Path to zarr dataset")
    parser.add_argument("--num-envs", type=int, default=32,
                       help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=300,
                       help="Maximum steps per environment")
    parser.add_argument("--pred-horizon", type=int, 
                       default=config.get('pred_horizon', 16) if config else 16,
                       help="Prediction horizon")
    parser.add_argument("--obs-horizon", type=int, 
                       default=config.get('obs_horizon', 2) if config else 2,
                       help="Observation horizon")
    parser.add_argument("--action-horizon", type=int, 
                       default=config.get('action_horizon', 8) if config else 8,
                       help="Action horizon")
    parser.add_argument("--action-dim", type=int, 
                       default=config.get('action_dim', 2) if config else 2,
                       help="Action dimension")
    parser.add_argument("--obs-dim", type=int, 
                       default=config.get('obs_dim', 5) if config else 5,
                       help="Observation dimension")
    parser.add_argument("--fm-timesteps", type=int, 
                       default=config.get('fm_timesteps', 100) if config else 100,
                       help="Number of timesteps for flow matching inference")
    parser.add_argument("--sigma", type=float, 
                       default=config.get('sigma', 0.0) if config else 0.0,
                       help="Noise scale for flow matching paths")
    parser.add_argument("--use-position-decoder", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=config.get('use_position_decoder', True) if config else True,
                       help="Use position decoder")
    parser.add_argument("--position-decoder-down-dims", type=int, nargs="+",
                       default=config.get('position_decoder_down_dims', [256]) if config else [256],
                       help="Channel dimensions for position decoder U-Net")
    parser.add_argument("--position-decoder-n-groups", type=int,
                       default=config.get('position_decoder_n_groups', 4) if config else 4,
                       help="Number of groups for position decoder GroupNorm")
    parser.add_argument("--position-decoder-fm-timesteps", type=int,
                       default=config.get('position_decoder_fm_timesteps', 8) if config else 8,
                       help="Number of timesteps for position decoder inference")
    parser.add_argument("--position-decoder-num-particles", type=int,
                       default=config.get('position_decoder_num_particles', 1) if config else 1,
                       help="Number of particles for position prediction")
    parser.add_argument("--position-decoder-particles-aggregation", type=str,
                       default=config.get('position_decoder_particles_aggregation', 'median') if config else 'median',
                       choices=["median", "knn"],
                       help="Method to aggregate particles when num_particles > 1. "
                            "Options: 'median' (element-wise median) or 'knn' (KNN-based density estimation).")
    parser.add_argument("--share-noise", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=config.get('share_noise', False) if config else False,
                       help="Share noise between position and action predictions")
    parser.add_argument("--shared-noise-base", type=str,
                       default=config.get('shared_noise_base', 'action') if config else 'action',
                       help="Base for noise sharing")
    parser.add_argument("--use-gt-reference-for-local-policy", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=config.get('use_gt_reference_for_local_policy', False) if config else False,
                       help="Use ground truth reference position for local policy conditioning")
    parser.add_argument("--use-film-conditioning", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=config.get('use_film_conditioning', False) if config else False,
                       help="Use FiLM (Feature-wise Linear Modulation) for position conditioning")
    parser.add_argument("--film-hidden-dim", type=int,
                       default=config.get('film_hidden_dim', 32) if config else 32,
                       help="Hidden dimension for FiLM position encoder MLP")
    parser.add_argument("--film-predict-scale", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=config.get('film_predict_scale', True) if config else True,
                       help="FiLM predicts both scale and bias (full FiLM)")
    parser.add_argument("--eval-input-noise-std", type=float, default=0.0,
                       help="Standard deviation of Gaussian noise to add to observations during evaluation. "
                            "This is separate from the model's input_noise_std parameter (which should be 0 for evaluation). "
                            "Applied to both pose estimator and local policy inputs. Default: 0.0 (no noise).")
    parser.add_argument("--disable-reference-conditioning", action='store_true',
                       default=False,
                       help="Disable reference position conditioning for the local policy (ablation study). "
                            "Default: False (for backward compatibility with old models).")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu), auto-detect if None")
    parser.add_argument("--out-path", type=str, default=None,
                       help="Output path for evaluation video (auto-generated if None)")
    
    # Parse with the checkpoint path and any remaining args
    args = parser.parse_args(["--ckpt-path", initial_args.ckpt_path] + remaining_args)
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model type: LocalFlowPolicy2D")
    
    # Print hyperparameters for verification
    print("\n" + "="*50)
    print("Model Hyperparameters (for verification):")
    print("="*50)
    print(f"  act_dim: {args.action_dim}")
    print(f"  obs_horizon: {args.obs_horizon}")
    print(f"  act_horizon: {args.action_horizon}")
    print(f"  pred_horizon: {args.pred_horizon}")
    print(f"  obs_dim: {args.obs_dim}")
    print(f"  sigma: {args.sigma}")
    print(f"  fm_timesteps: {args.fm_timesteps}")
    print(f"  use_position_decoder: {args.use_position_decoder}")
    print(f"  position_decoder_down_dims: {args.position_decoder_down_dims}")
    print(f"  position_decoder_n_groups: {args.position_decoder_n_groups}")
    print(f"  position_decoder_fm_timesteps: {args.position_decoder_fm_timesteps}")
    print(f"  position_decoder_num_particles: {args.position_decoder_num_particles}")
    print(f"  position_decoder_particles_aggregation: {args.position_decoder_particles_aggregation}")
    print(f"  share_noise: {args.share_noise}")
    print(f"  shared_noise_base: {args.shared_noise_base}")
    print(f"  use_gt_reference_for_local_policy: {args.use_gt_reference_for_local_policy}")
    print(f"  use_film_conditioning: {args.use_film_conditioning}")
    print(f"  film_hidden_dim: {args.film_hidden_dim}")
    print(f"  film_predict_scale: {args.film_predict_scale}")
    # Handle disable_reference_conditioning: use config default if flag was not explicitly set
    # When action='store_true', if flag is not present, value is False
    # If flag is present, value is True (explicit override)
    # If config exists and has the parameter, use it as default (unless flag was set)
    if config and 'disable_reference_conditioning' in config:
        # If flag was not explicitly set (value is False), use config value
        if not args.disable_reference_conditioning:
            args.disable_reference_conditioning = config.get('disable_reference_conditioning', False)
    
    # Ensure backward compatibility: old models (without disable_reference_conditioning in config)
    # should be evaluated with disable_reference_conditioning=False
    if config and 'disable_reference_conditioning' not in config:
        if args.disable_reference_conditioning:
            print("WARNING: You are evaluating an old model (trained with reference conditioning)")
            print("         with disable_reference_conditioning=True. This is an ablation study.")
            print("         For standard evaluation, use disable_reference_conditioning=False.\n")
        else:
            print("INFO: Old model detected (no disable_reference_conditioning in config).")
            print("      Using disable_reference_conditioning=False for backward compatibility.\n")
        # Explicitly set to False for old models to ensure consistency
        args.disable_reference_conditioning = False
    
    print(f"  disable_reference_conditioning: {args.disable_reference_conditioning} (should be False for old models)")
    print(f"  eval_input_noise_std: {args.eval_input_noise_std} (evaluation-time noise, always 0.0)")
    print("="*50 + "\n")
    
    # Load dataset for stats
    # IMPORTANT: use the same segmented dataset class as in training so that
    # normalization statistics (min / max) match the ones used during training.
    # This keeps evaluation here consistent with the on-the-fly evaluations
    # performed inside train_local_flow_2d.py.
    dataset = PushTSegmentedDatasetSimple(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_demos=config.get('max_demos', None) if config else None,
        contact_threshold=config.get('contact_threshold', 0.1) if config else 0.1,
        min_segment_length=config.get('min_segment_length', 5) if config else 5,
    )
    stats = dataset.stats
    
    # Create model
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
        position_decoder_num_particles=args.position_decoder_num_particles,
        # position_decoder_particles_aggregation=args.position_decoder_particles_aggregation,
        share_noise=args.share_noise,
        shared_noise_base=args.shared_noise_base,
        use_gt_reference_for_local_policy=args.use_gt_reference_for_local_policy,
        use_film_conditioning=args.use_film_conditioning,
        film_hidden_dim=args.film_hidden_dim,
        film_predict_scale=args.film_predict_scale,
        # input_noise_std=0.0,  # Always 0 for evaluation (model parameter, not evaluation-time noise)
        disable_reference_conditioning=args.disable_reference_conditioning,
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    # Use strict=False to handle old checkpoints that might not have all new parameters
    # This is safe because disable_reference_conditioning is just a flag, not a model parameter
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint (this is normal for old models): {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    model.eval()
    
    # Determine output path
    if args.out_path is None:
        # Extract the experiment directory from checkpoint path
        # ckpt_path: log/dp/exp_name/checkpoints/best_ema_model.pt
        # We want: log/dp/exp_name/eval_results/eval.mp4
        # Note: exp_name includes aggregation method (median/knn) from training
        # Format: sd-pusht-local-flow-2d_V3-demos-{N}-seed{S}-{noise}-{ref}-{film}-{agg}
        ckpt_dir = os.path.dirname(args.ckpt_path)  # log/dp/exp_name/checkpoints
        exp_dir = os.path.dirname(ckpt_dir)  # log/dp/exp_name
        eval_results_dir = os.path.join(exp_dir, "eval_results")
        os.makedirs(eval_results_dir, exist_ok=True)
        out_path = os.path.join(eval_results_dir, "eval.mp4")
    else:
        out_path = args.out_path
        # Create directory if it doesn't exist
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving evaluation video to: {out_path}")
    
    # Evaluate
    # IMPORTANT: Always set eval_input_noise_std=0.0 for proper evaluation
    # (even if model was trained with noise, we evaluate without noise)
    # The eval_input_noise_std parameter is for robustness testing, not standard evaluation
    eval_input_noise_std_for_eval = 0.0  # Always 0.0 for standard evaluation
    
    results = evaluate_local_flow_2d(
        model=model,
        stats=stats,
        out_path=out_path,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        device=device,
        save_video=True,
        # eval_input_noise_std=eval_input_noise_std_for_eval,  # Always 0.0 for standard evaluation
    )
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Success Rate: {results['success_rate']:.2%} ({results['success_count']}/{args.num_envs})")
    print(f"Mean Cumulative Score: {results['mean_score']:.4f}")
    print(f"Max Cumulative Score: {results['max_score']:.4f}")
    print(f"Mean Max Single-Step Reward (avg of max per env): {results['mean_max_single_reward']:.4f}")
    print(f"Steps Done: {results['steps_done']}")
    print(f"Video saved to: {out_path}")
    print("="*50)
    
    # Determine output directory (same as video directory)
    if args.out_path is None:
        ckpt_dir = os.path.dirname(args.ckpt_path)
        exp_dir = os.path.dirname(ckpt_dir)
        eval_results_dir = os.path.join(exp_dir, "eval_results")
    else:
        eval_results_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else "."
    
    # Save results as text file
    txt_path = os.path.join(eval_results_dir, "eval_results.txt")
    with open(txt_path, "w") as f:
        f.write("="*50 + "\n")
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.ckpt_path}\n")
        f.write(f"Model Type: LocalFlowPolicy2D\n")
        f.write(f"\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  Number of Environments: {args.num_envs}\n")
        f.write(f"  Max Steps: {args.max_steps}\n")
        f.write(f"  Prediction Horizon: {args.pred_horizon}\n")
        f.write(f"  Observation Horizon: {args.obs_horizon}\n")
        f.write(f"  Action Horizon: {args.action_horizon}\n")
        f.write(f"  Flow Matching Timesteps: {args.fm_timesteps}\n")
        f.write(f"  Sigma: {args.sigma}\n")
        f.write(f"  Use Position Decoder: {args.use_position_decoder}\n")
        f.write(f"  Position Decoder Num Particles: {args.position_decoder_num_particles}\n")
        f.write(f"  Position Decoder Particles Aggregation: {args.position_decoder_particles_aggregation}\n")
        f.write(f"  Share Noise: {args.share_noise}\n")
        f.write(f"  Use GT Reference for Local Policy: {args.use_gt_reference_for_local_policy}\n")
        f.write(f"  Use FiLM Conditioning: {args.use_film_conditioning}\n")
        f.write(f"  FiLM Hidden Dim: {args.film_hidden_dim}\n")
        f.write(f"  FiLM Predict Scale: {args.film_predict_scale}\n")
        f.write(f"  Eval Input Noise Std: {args.eval_input_noise_std}\n")
        f.write(f"\n")
        f.write(f"Results:\n")
        f.write(f"  Success Rate: {results['success_rate']:.4f} ({results['success_rate']:.2%})\n")
        f.write(f"  Success Count: {results['success_count']}/{args.num_envs}\n")
        f.write(f"  Mean Cumulative Score: {results['mean_score']:.4f}\n")
        f.write(f"  Max Cumulative Score: {results['max_score']:.4f}\n")
        f.write(f"  Mean Max Single-Step Reward (avg of max per env): {results['mean_max_single_reward']:.4f}\n")
        f.write(f"  Steps Done: {results['steps_done']}\n")
        f.write(f"\n")
        f.write(f"Output Files:\n")
        f.write(f"  Video: {out_path}\n")
        f.write(f"  Results Text: {txt_path}\n")
        f.write(f"  Results NPZ: {os.path.join(eval_results_dir, 'eval_results.npz')}\n")
        f.write("="*50 + "\n")
    
    print(f"Results saved to text file: {txt_path}")
    
    # Save results as npz file
    npz_path = os.path.join(eval_results_dir, "eval_results.npz")
    np.savez(
        npz_path,
        success_rate=results['success_rate'],
        success_count=results['success_count'],
        num_envs=args.num_envs,
        mean_score=results['mean_score'],
        max_score=results['max_score'],
        mean_max_single_reward=results['mean_max_single_reward'],
        steps_done=results['steps_done'],
        model_type=np.array(['local_flow_2d'], dtype=object),
        checkpoint_path=np.array([args.ckpt_path], dtype=object),
        video_path=np.array([out_path], dtype=object),
        num_envs_param=args.num_envs,
        max_steps=args.max_steps,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        fm_timesteps=args.fm_timesteps,
        sigma=args.sigma,
        use_position_decoder=args.use_position_decoder,
        position_decoder_num_particles=args.position_decoder_num_particles,
        position_decoder_particles_aggregation=args.position_decoder_particles_aggregation,
        share_noise=args.share_noise,
        use_gt_reference_for_local_policy=args.use_gt_reference_for_local_policy,
        use_film_conditioning=args.use_film_conditioning,
        film_hidden_dim=args.film_hidden_dim,
        film_predict_scale=args.film_predict_scale,
        disable_reference_conditioning=args.disable_reference_conditioning,
        eval_input_noise_std=args.eval_input_noise_std,
        timestamp=np.array([datetime.now().isoformat()], dtype=object)
    )
    
    print(f"Results saved to npz file: {npz_path}")

