#!/usr/bin/env python3
"""Ablation evaluation script for local flow 2D policy on PushT.

This script:
1. Finds all trained models with KNN aggregation method
2. For each model, runs evaluation with different combinations of:
   - Number of particles (num_particles)
   - Number of clusters (k_clusters for KNN aggregation)

Example usage:
python SD_pusht/scripts/eval_local_flow_2d_ablation.py \
    --base-dir log/dp \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --num-envs 32 \
    --max-steps 300 \
    --num-particles-list 1 4 8 16 32 \
    --k-clusters-list 1 2 4 8
"""

import argparse
import os
import json
import glob
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from SD_pusht.models import LocalFlowPolicy2D
from SD_pusht.datasets import PushTSegmentedDatasetSimple
from SD_pusht.utils.evaluation import evaluate_local_flow_2d


def load_config_from_checkpoint(ckpt_path):
    """
    Load config.json from the experiment directory.
    Returns None if config file doesn't exist.
    """
    ckpt_dir = os.path.dirname(ckpt_path)  # log/dp/exp_name/checkpoints
    exp_dir = os.path.dirname(ckpt_dir)  # log/dp/exp_name
    config_path = os.path.join(exp_dir, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        return None


def find_knn_models(base_dir, checkpoint_name="best_ema_model.pt"):
    """
    Find all model checkpoints trained with KNN aggregation method.
    
    Args:
        base_dir: Base directory containing experiment folders (e.g., "log/dp")
        checkpoint_name: Name of checkpoint file to look for
    
    Returns:
        List of tuples: (checkpoint_path, config, experiment_name)
    """
    knn_models = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Warning: Base directory {base_dir} does not exist!")
        return knn_models
    
    # Find all experiment directories
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check for checkpoint
        ckpt_path = exp_dir / "checkpoints" / checkpoint_name
        if not ckpt_path.exists():
            continue
        
        # Load config to check aggregation method
        config = load_config_from_checkpoint(str(ckpt_path))
        if config is None:
            # Try to infer from experiment name
            exp_name = exp_dir.name
            if "knn" in exp_name.lower():
                print(f"Found KNN model (by name): {exp_name}")
                knn_models.append((str(ckpt_path), None, exp_name))
            continue
        
        # Check if aggregation method is KNN
        aggregation = config.get('position_decoder_particles_aggregation', 'median')
        if aggregation == 'knn':
            print(f"Found KNN model: {exp_dir.name}")
            knn_models.append((str(ckpt_path), config, exp_dir.name))
    
    return knn_models


def evaluate_with_ablation(
    ckpt_path,
    config,
    exp_name,
    dataset_path,
    num_envs,
    max_steps,
    num_particles_list,
    k_clusters_list,
    device,
    base_output_dir,
    model_config_overrides=None,
):
    """
    Evaluate a model with different combinations of num_particles and k_clusters.
    
    Args:
        ckpt_path: Path to checkpoint
        config: Model config (or None)
        exp_name: Experiment name
        dataset_path: Path to dataset
        num_envs: Number of parallel environments
        max_steps: Maximum steps per environment
        num_particles_list: List of num_particles values to test
        k_clusters_list: List of k_clusters values to test
        device: Device to run on
        base_output_dir: Base directory for output results
        model_config_overrides: Dict of model config overrides (None values mean use config/default)
    
    Returns:
        Dictionary with results for all combinations
    """
    print(f"\n{'='*80}")
    print(f"Evaluating model: {exp_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*80}\n")
    
    # Load config or use defaults
    if config is None:
        print("Warning: No config found, using defaults from checkpoint path parsing")
        # Try to extract from exp_name
        # Format: sd-pusht-local-flow-2d_V3-demos-{N}-seed{S}-{noise}-{ref}-{film}-{agg}
        config = {}
    
    # Apply model config overrides (command-line args override config)
    if model_config_overrides is None:
        model_config_overrides = {}
    
    # Helper function to get config value with override
    def get_config(key, default):
        if key in model_config_overrides and model_config_overrides[key] is not None:
            return model_config_overrides[key]
        return config.get(key, default)
    
    
    # Load dataset for stats
    dataset = PushTSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=get_config('pred_horizon', 16),
        obs_horizon=get_config('obs_horizon', 2),
        action_horizon=get_config('action_horizon', 8),
        max_demos=get_config('max_demos', None),
        contact_threshold=get_config('contact_threshold', 0.1),
        min_segment_length=get_config('min_segment_length', 5),
    )
    stats = dataset.stats
    
    # Create model with config values (overridden by command-line args if provided)
    # Use same model creation logic as eval_local_flow_2d.py
    model = LocalFlowPolicy2D(
        act_dim=get_config('action_dim', 2),
        obs_horizon=get_config('obs_horizon', 2),
        act_horizon=get_config('action_horizon', 8),
        pred_horizon=get_config('pred_horizon', 16),
        obs_dim=get_config('obs_dim', 5),
        sigma=get_config('sigma', 0.0),
        fm_timesteps=get_config('fm_timesteps', 100),
        use_position_decoder=get_config('use_position_decoder', True),
        position_decoder_down_dims=get_config('position_decoder_down_dims', [256]),
        position_decoder_n_groups=get_config('position_decoder_n_groups', 4),
        position_decoder_fm_timesteps=get_config('position_decoder_fm_timesteps', 8),
        position_decoder_num_particles=get_config('position_decoder_num_particles', 1),
        position_decoder_particles_aggregation='knn',  # Force KNN for ablation
        share_noise=get_config('share_noise', False),
        shared_noise_base=get_config('shared_noise_base', 'action'),
        use_gt_reference_for_local_policy=get_config('use_gt_reference_for_local_policy', False),
        use_film_conditioning=get_config('use_film_conditioning', False),
        film_hidden_dim=get_config('film_hidden_dim', 32),
        film_predict_scale=get_config('film_predict_scale', True),
        disable_reference_conditioning=get_config('disable_reference_conditioning', False),
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    # Use strict=False to handle old checkpoints that might not have all new parameters
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint (this is normal for old models): {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    model.eval()
    
    # Store all results
    all_results = {}
    
    # Evaluate all combinations
    for num_particles in num_particles_list:
        for k_clusters in k_clusters_list:
            # Skip invalid combinations
            if k_clusters >= num_particles:
                print(f"Skipping: k_clusters={k_clusters} >= num_particles={num_particles}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Testing: num_particles={num_particles}, k_clusters={k_clusters}")
            print(f"{'='*60}")
            
            # Temporarily modify model's num_particles
            original_num_particles = model.position_decoder.num_particles
            model.position_decoder.num_particles = num_particles
            
            # Create output directory for this combination
            output_dir = Path(base_output_dir) / exp_name / "ablation" / f"particles_{num_particles}_clusters_{k_clusters}"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "eval.mp4"
            
            try:
                # Create particles visualization directory
                particles_dir = output_dir / "particles"
                particles_dir.mkdir(parents=True, exist_ok=True)
                
                # Run evaluation with k_clusters parameter and particle visualization
                # Use get_config helper function defined above
                results = evaluate_local_flow_2d(
                    model=model,
                    stats=stats,
                    out_path=str(out_path),
                    num_envs=num_envs,
                    max_steps=max_steps,
                    pred_horizon=get_config('pred_horizon', 16),
                    obs_horizon=get_config('obs_horizon', 2),
                    action_horizon=get_config('action_horizon', 8),
                    device=device,
                    save_video=True,
                    k_clusters=k_clusters,
                    visualize_particles=True,
                    particles_output_dir=str(particles_dir),
                )
                
                # Store results
                key = f"particles_{num_particles}_clusters_{k_clusters}"
                all_results[key] = {
                    'num_particles': num_particles,
                    'k_clusters': k_clusters,
                    'success_rate': results['success_rate'],
                    'success_count': results['success_count'],
                    'mean_score': results['mean_score'],
                    'max_score': results['max_score'],
                    'mean_max_single_reward': results['mean_max_single_reward'],
                    'steps_done': results['steps_done'],
                    'video_path': str(out_path),
                }
                
                print(f"\nResults for particles={num_particles}, clusters={k_clusters}:")
                print(f"  Success Rate: {results['success_rate']:.2%}")
                print(f"  Mean Score: {results['mean_score']:.4f}")
                print(f"  Max Score: {results['max_score']:.4f}")
                
            except Exception as e:
                print(f"Error evaluating particles={num_particles}, clusters={k_clusters}: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Restore original num_particles
                model.position_decoder.num_particles = original_num_particles
    
    return all_results


def save_ablation_results(all_results, exp_name, output_dir):
    """Save ablation results to files and generate plots."""
    output_path = Path(output_dir) / exp_name / "ablation" / "ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for key, value in all_results.items():
        results_serializable[key] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in value.items()
        }
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nSaved ablation results to: {output_path}")
    
    # Also save as CSV for easy analysis
    csv_path = output_path.parent / "ablation_results.csv"
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['num_particles', 'k_clusters', 'success_rate', 
                                               'success_count', 'mean_score', 'max_score',
                                               'mean_max_single_reward', 'steps_done'])
        writer.writeheader()
        for key, result in all_results.items():
            row = {k: v for k, v in result.items() if k != 'video_path'}
            writer.writerow(row)
    
    print(f"Saved CSV results to: {csv_path}")
    
    # Generate plots: performance vs num_particles for each fixed k_clusters
    plot_ablation_results(all_results, exp_name, output_dir)


def plot_ablation_results(all_results, exp_name, output_dir):
    """Plot performance vs num_particles for each fixed k_clusters value."""
    import matplotlib.pyplot as plt
    
    # Group results by k_clusters
    results_by_k = {}
    for key, result in all_results.items():
        k = result['k_clusters']
        if k not in results_by_k:
            results_by_k[k] = []
        results_by_k[k].append(result)
    
    # Create plots directory
    plots_dir = Path(output_dir) / exp_name / "ablation" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot metrics vs num_particles for each k_clusters
    metrics = ['success_rate', 'mean_score', 'max_score', 'mean_max_single_reward']
    metric_labels = {
        'success_rate': 'Success Rate',
        'mean_score': 'Mean Score',
        'max_score': 'Max Score',
        'mean_max_single_reward': 'Mean Max Single-Step Reward'
    }
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot a line for each k_clusters value
        for k in sorted(results_by_k.keys()):
            results_k = results_by_k[k]
            # Sort by num_particles
            results_k_sorted = sorted(results_k, key=lambda x: x['num_particles'])
            
            num_particles_list = [r['num_particles'] for r in results_k_sorted]
            metric_values = [r[metric] for r in results_k_sorted]
            
            ax.plot(num_particles_list, metric_values, marker='o', label=f'k_clusters={k}', linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Particles', fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f'{metric_labels[metric]} vs Number of Particles\n(Experiment: {exp_name})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)  # Log scale for num_particles
        
        # Save plot
        plot_path = plots_dir / f"{metric}_vs_particles.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {plot_path}")
    
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation evaluation for local flow 2D policy with KNN aggregation"
    )
    parser.add_argument("--base-dir", type=str, default="log/dp",
                       help="Base directory containing experiment folders")
    parser.add_argument("--dataset-path", type=str, 
                       default="datasets/pusht_cchi_v7_replay.zarr.zip",
                       help="Path to zarr dataset")
    parser.add_argument("--num-envs", type=int, default=32,
                       help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=300,
                       help="Maximum steps per environment")
    parser.add_argument("--num-particles-list", type=int, nargs="+",
                       default=[1, 4, 8, 16, 32],
                       help="List of num_particles values to test")
    parser.add_argument("--k-clusters-list", type=int, nargs="+",
                       default=[1, 2, 4, 8],
                       help="List of k_clusters values to test")
    parser.add_argument("--checkpoint-name", type=str, default="best_ema_model.pt",
                       help="Name of checkpoint file to look for")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu), auto-detect if None")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for results (default: same as base-dir)")
    
    # Model configuration arguments (same as eval_local_flow_2d.py)
    # These will override config.json values if provided
    parser.add_argument("--pred-horizon", type=int, default=None,
                       help="Prediction horizon (overrides config)")
    parser.add_argument("--obs-horizon", type=int, default=None,
                       help="Observation horizon (overrides config)")
    parser.add_argument("--action-horizon", type=int, default=None,
                       help="Action horizon (overrides config)")
    parser.add_argument("--action-dim", type=int, default=None,
                       help="Action dimension (overrides config)")
    parser.add_argument("--obs-dim", type=int, default=None,
                       help="Observation dimension (overrides config)")
    parser.add_argument("--fm-timesteps", type=int, default=None,
                       help="Number of timesteps for flow matching inference (overrides config)")
    parser.add_argument("--sigma", type=float, default=None,
                       help="Noise scale for flow matching paths (overrides config)")
    parser.add_argument("--use-position-decoder", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=None,
                       help="Use position decoder (overrides config)")
    parser.add_argument("--position-decoder-down-dims", type=int, nargs="+", default=None,
                       help="Channel dimensions for position decoder U-Net (overrides config)")
    parser.add_argument("--position-decoder-n-groups", type=int, default=None,
                       help="Number of groups for position decoder GroupNorm (overrides config)")
    parser.add_argument("--position-decoder-fm-timesteps", type=int, default=None,
                       help="Number of timesteps for position decoder inference (overrides config)")
    parser.add_argument("--share-noise", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=None,
                       help="Share noise between position and action predictions (overrides config)")
    parser.add_argument("--shared-noise-base", type=str, default=None,
                       help="Base for noise sharing (overrides config)")
    parser.add_argument("--use-gt-reference-for-local-policy", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=None,
                       help="Use ground truth reference position for local policy conditioning (overrides config)")
    parser.add_argument("--use-film-conditioning", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=None,
                       help="Use FiLM conditioning (overrides config)")
    parser.add_argument("--film-hidden-dim", type=int, default=None,
                       help="Hidden dimension for FiLM position encoder MLP (overrides config)")
    parser.add_argument("--film-predict-scale", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=None,
                       help="FiLM predicts both scale and bias (overrides config)")
    parser.add_argument("--disable-reference-conditioning", type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
                       default=None,
                       help="Disable reference position conditioning for the local policy (overrides config)")
    parser.add_argument("--eval-input-noise-std", type=float, default=0.0,
                       help="Standard deviation of Gaussian noise to add to observations during evaluation. "
                            "This is separate from the model's input_noise_std parameter. Default: 0.0 (no noise).")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set output directory
    if args.output_dir is None:
        output_dir = args.base_dir
    else:
        output_dir = args.output_dir
    
    # Find all KNN models
    print("\n" + "="*80)
    print("Searching for KNN-trained models...")
    print("="*80)
    knn_models = find_knn_models(args.base_dir, args.checkpoint_name)
    
    if len(knn_models) == 0:
        print("No KNN models found! Exiting.")
        exit(1)
    
    print(f"\nFound {len(knn_models)} KNN model(s)")
    
    # Build model config overrides dict from command-line args
    model_config_overrides = {
        'pred_horizon': args.pred_horizon,
        'obs_horizon': args.obs_horizon,
        'action_horizon': args.action_horizon,
        'action_dim': args.action_dim,
        'obs_dim': args.obs_dim,
        'fm_timesteps': args.fm_timesteps,
        'sigma': args.sigma,
        'use_position_decoder': args.use_position_decoder,
        'position_decoder_down_dims': args.position_decoder_down_dims,
        'position_decoder_n_groups': args.position_decoder_n_groups,
        'position_decoder_fm_timesteps': args.position_decoder_fm_timesteps,
        'share_noise': args.share_noise,
        'shared_noise_base': args.shared_noise_base,
        'use_gt_reference_for_local_policy': args.use_gt_reference_for_local_policy,
        'use_film_conditioning': args.use_film_conditioning,
        'film_hidden_dim': args.film_hidden_dim,
        'film_predict_scale': args.film_predict_scale,
        'disable_reference_conditioning': args.disable_reference_conditioning,
    }
    
    # Evaluate each model
    all_model_results = {}
    for ckpt_path, config, exp_name in knn_models:
        try:
            results = evaluate_with_ablation(
                ckpt_path=ckpt_path,
                config=config,
                exp_name=exp_name,
                dataset_path=args.dataset_path,
                num_envs=args.num_envs,
                max_steps=args.max_steps,
                num_particles_list=args.num_particles_list,
                k_clusters_list=args.k_clusters_list,
                device=device,
                base_output_dir=output_dir,
                model_config_overrides=model_config_overrides,
            )
            all_model_results[exp_name] = results
            save_ablation_results(results, exp_name, output_dir)
        except Exception as e:
            print(f"Error evaluating model {exp_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Ablation evaluation complete!")
    print("="*80)
    print(f"\nEvaluated {len(all_model_results)} model(s)")
    print(f"Results saved to: {output_dir}")
