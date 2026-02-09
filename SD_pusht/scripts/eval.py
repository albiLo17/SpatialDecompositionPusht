#!/usr/bin/env python3
"""Evaluation script for diffusion/flow matching policy on PushT."""
"""
Example usage:
python SD_pusht/scripts/eval.py \
    --ckpt-path log/dp/sd-pusht-diffusion-demos-100-seed0/checkpoints/ema_diffusion_model.pt \
    --use-flow-matching \
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

from SD_pusht.models import Diffusion, FlowMatching
from SD_pusht.datasets import PushTStateDataset
from SD_pusht.utils.evaluation import evaluate_model


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


if __name__ == "__main__":
    # First parse args minimally to get checkpoint path
    parser = argparse.ArgumentParser(description="Evaluate diffusion/flow matching policy on PushT")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to checkpoint file")
    initial_args, remaining_args = parser.parse_known_args()
    
    # Try to load config from experiment directory
    config = load_config_from_checkpoint(initial_args.ckpt_path)
    
    # Now parse full args, using config as defaults if available
    parser = argparse.ArgumentParser(description="Evaluate diffusion/flow matching policy on PushT")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--use-flow-matching", action="store_true",
                       help="Use flow matching model (otherwise uses diffusion)")
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
    parser.add_argument("--num-diffusion-iters", type=int, 
                       default=config.get('num_diffusion_iters', 100) if config else 100,
                       help="Number of diffusion iterations (for diffusion model)")
    parser.add_argument("--fm-timesteps", type=int, 
                       default=config.get('fm_timesteps', 100) if config else 100,
                       help="Number of timesteps for flow matching inference")
    parser.add_argument("--sigma", type=float, 
                       default=config.get('sigma', 0.0) if config else 0.0,
                       help="Noise scale for flow matching paths")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu), auto-detect if None")
    parser.add_argument("--out-path", type=str, default=None,
                       help="Output path for evaluation video (auto-generated if None)")
    
    # Parse with the checkpoint path and any remaining args
    args = parser.parse_args(["--ckpt-path", initial_args.ckpt_path] + remaining_args)
    
    # Handle boolean flag from config if not explicitly set via command line
    if config and '--use-flow-matching' not in remaining_args:
        args.use_flow_matching = config.get('use_flow_matching', False)
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model type: {'Flow Matching' if args.use_flow_matching else 'Diffusion'}")
    
    # Load dataset for stats
    dataset = PushTStateDataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
    )
    stats = dataset.stats
    
    # Create model
    if args.use_flow_matching:
        model = FlowMatching(
            act_dim=args.action_dim,
            obs_horizon=args.obs_horizon,
            act_horizon=args.action_horizon,
            pred_horizon=args.pred_horizon,
            obs_dim=args.obs_dim,
            sigma=args.sigma,
            fm_timesteps=args.fm_timesteps
        ).to(device)
    else:
        model = Diffusion(
            act_dim=args.action_dim,
            obs_horizon=args.obs_horizon,
            act_horizon=args.action_horizon,
            pred_horizon=args.pred_horizon,
            obs_dim=args.obs_dim,
            num_diffusion_iters=args.num_diffusion_iters
        ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Determine output path
    if args.out_path is None:
        # Extract the experiment directory from checkpoint path
        # ckpt_path: log/dp/exp_name/checkpoints/ema_model.pt
        # We want: log/dp/exp_name/eval_results/eval.mp4
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
    results = evaluate_model(
        model=model,
        stats=stats,
        out_path=out_path,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        device=device,
        use_flow_matching=args.use_flow_matching,
        save_video=True
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
        f.write(f"Model Type: {'Flow Matching' if args.use_flow_matching else 'Diffusion'}\n")
        f.write(f"\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  Number of Environments: {args.num_envs}\n")
        f.write(f"  Max Steps: {args.max_steps}\n")
        f.write(f"  Prediction Horizon: {args.pred_horizon}\n")
        f.write(f"  Observation Horizon: {args.obs_horizon}\n")
        f.write(f"  Action Horizon: {args.action_horizon}\n")
        if args.use_flow_matching:
            f.write(f"  Flow Matching Timesteps: {args.fm_timesteps}\n")
            f.write(f"  Sigma: {args.sigma}\n")
        else:
            f.write(f"  Diffusion Iterations: {args.num_diffusion_iters}\n")
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
        max_score=results['max_score'],  # Max cumulative reward
        mean_max_single_reward=results['mean_max_single_reward'],  # Average of max single-step reward per environment
        steps_done=results['steps_done'],
        model_type=np.array(['flow_matching' if args.use_flow_matching else 'diffusion'], dtype=object),
        checkpoint_path=np.array([args.ckpt_path], dtype=object),
        video_path=np.array([out_path], dtype=object),
        num_envs_param=args.num_envs,
        max_steps=args.max_steps,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        num_diffusion_iters=args.num_diffusion_iters if not args.use_flow_matching else 0,
        fm_timesteps=args.fm_timesteps if args.use_flow_matching else 0,
        sigma=args.sigma if args.use_flow_matching else 0.0,
        timestamp=np.array([datetime.now().isoformat()], dtype=object)
    )
    
    print(f"Results saved to npz file: {npz_path}")
