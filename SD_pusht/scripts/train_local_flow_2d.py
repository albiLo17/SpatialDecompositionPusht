#!/usr/bin/env python3
"""Training script for LocalFlowPolicy2D on PushT with segmented dataset."""

"""
python SD_pusht/scripts/train_local_flow_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 50 \
    --batch-size 256 \
    --epochs 2000 \
    --wandb \
    --use-position-decoder \
    --position-loss-coeff 1.0 \
    --contact-threshold 0.1 \
    --use-gt-reference-for-local-policy  # Optional: use GT reference position for local policy (oracle mode)
"""

import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import gdown

from SD_pusht.models import LocalFlowPolicy2D
from SD_pusht.datasets import PushTSegmentedDatasetSimple
from SD_pusht.utils.evaluation import evaluate_local_flow_2d, visualize_training_trajectory, visualize_action_predictions, visualize_position_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train LocalFlowPolicy2D on PushT")
    parser.add_argument("--dataset-path", type=str, 
                       default="datasets/pusht_cchi_v7_replay.zarr.zip",
                       help="Path to zarr dataset")
    parser.add_argument("--max-demos", type=int, default=None,
                       help="Maximum number of demonstration episodes to use for training")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--wandb", action="store_true", 
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="sd-pusht-local-flow-2d",
                       help="WandB project name")
    parser.add_argument("--eval-every", type=int, default=50,
                       help="Evaluate every N epochs (0 = no evaluation during training)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
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
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                       help="Weight decay")
    parser.add_argument("--ema-power", type=float, default=0.75,
                       help="EMA power parameter")
    parser.add_argument("--output-dir", type=str, default="log/dp",
                       help="Output directory for checkpoints and logs")
    # LocalFlowPolicy2D specific arguments
    parser.add_argument("--use-position-decoder", action="store_true", default=True,
                       help="Use position decoder")
    parser.add_argument("--position-decoder-down-dims", type=int, nargs="+", default=[256],
                       help="Channel dimensions for position decoder U-Net (use single level [256] for T=1)")
    parser.add_argument("--position-decoder-n-groups", type=int, default=4,
                       help="Number of groups for position decoder GroupNorm")
    parser.add_argument("--position-decoder-fm-timesteps", type=int, default=8,
                       help="Number of timesteps for position decoder inference")
    parser.add_argument("--position-decoder-num-particles", type=int, default=32,
                       help="Number of particles to sample for position prediction. "
                            "If > 1, samples multiple positions and aggregates them. Default: 32")
    parser.add_argument("--position-decoder-particles-aggregation", type=str, default="median",
                       choices=["median", "knn"],
                       help="Method to aggregate particles when num_particles > 1. "
                            "Options: 'median' (element-wise median) or 'knn' (KNN-based density estimation). "
                            "Default: 'median'.")
    parser.add_argument("--position-loss-coeff", type=float, default=1.0,
                       help="Coefficient for position loss")
    parser.add_argument("--share-noise", action="store_true",
                       help="Share noise between position and action predictions")
    parser.add_argument("--shared-noise-base", type=str, default="action",
                       choices=["action", "position", "combinatory"],
                       help="Base for noise sharing")
    parser.add_argument("--use-gt-reference-for-local-policy", action="store_true",
                       help="Use ground truth reference position for local policy conditioning "
                            "during training (oracle mode). This allows training the local policy "
                            "with privileged information to understand the best possible performance.")
    parser.add_argument("--use-film-conditioning", action="store_true", default=False,
                       help="Use FiLM (Feature-wise Linear Modulation) for position conditioning "
                            "instead of concatenation. Default: False (use concatenation).")
    parser.add_argument("--film-hidden-dim", type=int, default=64,
                       help="Hidden dimension for FiLM position encoder MLP (only used if --use-film-conditioning)")
    parser.add_argument("--film-predict-scale", action="store_true", default=True,
                       help="FiLM predicts both scale and bias (full FiLM). If False, only bias.")
    parser.add_argument("--input-noise-std", type=float, default=0.0,
                       help="Standard deviation of Gaussian noise to add to observations during training. "
                            "Applied to both pose estimator and local policy inputs. Default: 0.0 (no noise).")
    parser.add_argument("--disable-reference-conditioning", action="store_true",
                       help="Disable reference position conditioning for the local policy (ablation study). "
                            "Reference position will still be used for action transformation but not for conditioning.")
    # Dataset segmentation arguments
    # Note: cross-segment-padding is automatically computed as pred_horizon - action_horizon
    parser.add_argument("--contact-threshold", type=float, default=0.1,
                       help="Threshold for detecting block movement (contact)")
    parser.add_argument("--min-segment-length", type=int, default=5,
                       help="Minimum length for a valid segment")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Fraction of episodes to use for validation (0.0 = use remaining episodes, or some from training if not enough)")
    parser.add_argument("--val-min-episodes", type=int, default=10,
                       help="Minimum number of episodes for validation set (will use from training if needed)")
    parser.add_argument("--eval-sample-seed", type=int, default=None,
                       help="Random seed for selecting evaluation sample (None = use epoch number)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment name with configuration flags
    noise_str = "share-noise" if args.share_noise else "no-share-noise"
    ref_str = "gt-ref" if args.use_gt_reference_for_local_policy else "pred-ref"
    film_str = "film" if args.use_film_conditioning else "concat"
    agg_str = args.position_decoder_particles_aggregation  # "median" or "knn"
    input_noise_str = f"-noise{args.input_noise_std}" if args.input_noise_std > 0.0 else ""
    no_ref_cond_str = "-no-ref-cond" if args.disable_reference_conditioning else ""
    # V2 is with better dataset and shared encoder
    # exp_name = f"sd-pusht-local-flow-2d_V2-demos-{args.max_demos}-seed{args.seed}-{noise_str}-{ref_str}"
    # V3 is with the particles
    exp_name = f"sd-pusht-local-flow-2d_V3-demos-{args.max_demos}-seed{args.seed}-{noise_str}-{ref_str}-{film_str}-{agg_str}{input_noise_str}{no_ref_cond_str}"
    print(f"Experiment: {exp_name}")
    print(f"Using model: LocalFlowPolicy2D")
    print(f"Configuration: share_noise={args.share_noise}, use_gt_reference={args.use_gt_reference_for_local_policy}, use_film={args.use_film_conditioning}, particles_agg={args.position_decoder_particles_aggregation}, disable_ref_cond={args.disable_reference_conditioning}")
    
    # Create experiment directory and save config
    exp_dir = f"{args.output_dir}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    
    # Convert args to dict and save as JSON
    config_dict = vars(args).copy()
    # Convert any non-serializable objects to strings
    for key, value in config_dict.items():
        if not isinstance(value, (str, int, float, bool, type(None), list)):
            config_dict[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to: {config_path}")

    # optional wandb
    wandb_available = False
    if args.wandb:
        print("Using Weights & Biases logging")
        import wandb
        wandb_available = True
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.run.name = exp_name

    # download demonstration data from Google Drive if missing
    dataset_path = args.dataset_path
    if not os.path.isfile(dataset_path):
        print(f"Downloading dataset to {dataset_path}...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # Load dataset info to get total number of episodes
    import zarr
    dataset_root = zarr.open(dataset_path, 'r')
    all_episode_ends = dataset_root['meta']['episode_ends'][:]
    total_episodes = len(all_episode_ends)
    
    # create segmented training dataset from file (this will store demo_indices internally)
    dataset = PushTSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_demos=args.max_demos,  # Use max_demos as specified (keeps it fixed)
        use_contact_segmentation=True,
        contact_threshold=args.contact_threshold,
        min_segment_length=args.min_segment_length,
    )
    
    # Get demo indices used by training dataset
    train_demo_indices = getattr(dataset, 'demo_indices', None)
    if train_demo_indices is None:
        # Fallback: if demo_indices not stored, infer from max_demos
        train_demo_indices = list(range(args.max_demos if args.max_demos is not None else total_episodes))
    
    print(f"Training dataset uses {len(train_demo_indices)} episodes: {train_demo_indices[:5]}{'...' if len(train_demo_indices) > 5 else ''}")
    
    # Calculate validation demo indices
    val_dataset = None
    if args.val_split > 0:
        # Find remaining demo indices (not used in training)
        all_demo_indices = set(range(total_episodes))
        train_demo_indices_set = set(train_demo_indices)
        remaining_demo_indices = sorted(list(all_demo_indices - train_demo_indices_set))
        
        if len(remaining_demo_indices) >= args.val_min_episodes:
            # Use remaining episodes for validation
            val_demo_indices = remaining_demo_indices
            print(f"Using {len(val_demo_indices)} remaining episodes for validation: {val_demo_indices[:5]}{'...' if len(val_demo_indices) > 5 else ''}")
        else:
            # Not enough remaining episodes, randomly select some from training
            val_size = max(args.val_min_episodes, int(len(train_demo_indices) * args.val_split))
            rng = np.random.RandomState(args.seed)  # Use fixed seed for reproducibility
            val_demo_indices = sorted(rng.choice(train_demo_indices, size=val_size, replace=False).tolist())
            print(f"Not enough remaining episodes ({len(remaining_demo_indices)} < {args.val_min_episodes}). "
                  f"Randomly selected {len(val_demo_indices)} episodes from training for validation: {val_demo_indices[:5]}{'...' if len(val_demo_indices) > 5 else ''}")
        
        # Create validation dataset with specified demo indices
        try:
            val_dataset = PushTSegmentedDatasetSimple(
                dataset_path=dataset_path,
                pred_horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                action_horizon=args.action_horizon,
                demo_indices=val_demo_indices,  # Pass demo indices directly
                use_contact_segmentation=True,
                contact_threshold=args.contact_threshold,
                min_segment_length=args.min_segment_length,
            )
            print(f"Created validation dataset with {len(val_dataset)} samples from {len(val_demo_indices)} episodes")
        except Exception as e:
            print(f"Warning: Could not create validation dataset: {e}")
            print("Will use training dataset for visualization")
            val_dataset = None
    else:
        print("No validation split requested (--val-split=0)")

    # save training data statistics
    stats = getattr(dataset, "stats", None)
    
    # Create reference_pos_stats from obs stats (reference positions use same normalization as first 2 dims of obs)
    reference_pos_stats = None
    if stats and "obs" in stats:
        reference_pos_stats = {
            'min': stats["obs"]['min'][:2],  # First 2 dims are agent position
            'max': stats["obs"]['max'][:2],
        }

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape", batch['action'].shape)
    print("batch['reference_pos'].shape", batch['reference_pos'].shape)

    # create LocalFlowPolicy2D model
    agent = LocalFlowPolicy2D(
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
        position_decoder_particles_aggregation=args.position_decoder_particles_aggregation,
        position_loss_coeff=args.position_loss_coeff,
        share_noise=args.share_noise,
        shared_noise_base=args.shared_noise_base,
        use_gt_reference_for_local_policy=args.use_gt_reference_for_local_policy,
        use_film_conditioning=args.use_film_conditioning,
        film_hidden_dim=args.film_hidden_dim,
        film_predict_scale=args.film_predict_scale,
        # input_noise_std=args.input_noise_std,
        disable_reference_conditioning=args.disable_reference_conditioning,
    ).to(device)

    num_epochs = args.epochs

    # EMA
    ema = EMAModel(parameters=agent.parameters(), power=args.ema_power)

    optimizer = torch.optim.AdamW(
        params=agent.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    if wandb_available:
        try:
            wandb.watch(agent, log="all", log_freq=200)
        except Exception:
            pass

    # Track best evaluation metrics
    best_success_rate = -1.0
    best_epoch = -1
    best_model_path = None
    
    # Create checkpoint directory
    ckpt_dir = f"{args.output_dir}/{exp_name}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            epoch_action_loss = list()
            epoch_position_loss = list()
            
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    reference_pos = nbatch['reference_pos'].to(device)

                    # Compute loss using the model's compute_loss method
                    loss, loss_dict = agent.compute_loss(
                        obs_seq=nobs,
                        action_seq=naction,
                        reference_position=reference_pos,
                        action_stats=stats.get("action") if stats else None,
                        reference_pos_stats=reference_pos_stats,
                    )

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(agent.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    
                    # Track individual losses
                    action_loss = loss_dict.get("action", 0.0)
                    epoch_action_loss.append(action_loss)
                    
                    position_loss = loss_dict.get("position_total", 0.0)
                    if position_loss > 0:  # Only append if position decoder is used
                        epoch_position_loss.append(position_loss)
                    
                    tepoch.set_postfix(
                        loss=loss_cpu,
                        action_loss=action_loss,
                        position_loss=position_loss if position_loss > 0 else 0.0
                    )

            # Log at epoch level
            mean_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            mean_action_loss = np.mean(epoch_action_loss) if epoch_action_loss else 0.0
            mean_position_loss = np.mean(epoch_position_loss) if epoch_position_loss else 0.0
            
            tglobal.set_postfix(
                loss=mean_epoch_loss,
                action_loss=mean_action_loss,
                position_loss=mean_position_loss
            )
            
            if wandb_available:
                log_dict = {
                    "train/epoch_loss": mean_epoch_loss,
                    "train/action_loss": mean_action_loss,
                    "train/position_loss": mean_position_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else None,
                    "train/epoch": epoch_idx
                }
                wandb.log(log_dict)
            
            # evaluation step
            if args.eval_every > 0 and (epoch_idx + 1) % args.eval_every == 0:
                # Create EMA model for evaluation
                ema_model = LocalFlowPolicy2D(
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
                    position_decoder_particles_aggregation=args.position_decoder_particles_aggregation,
                    position_loss_coeff=args.position_loss_coeff,
                    share_noise=args.share_noise,
                    shared_noise_base=args.shared_noise_base,
                    use_gt_reference_for_local_policy=args.use_gt_reference_for_local_policy,
                    use_film_conditioning=args.use_film_conditioning,
                    film_hidden_dim=args.film_hidden_dim,
                    film_predict_scale=args.film_predict_scale,
                    # input_noise_std=args.input_noise_std,
                    disable_reference_conditioning=args.disable_reference_conditioning,
                ).to(device)
                
                ema.copy_to(ema_model.parameters())
                ema_model.eval()
                print(f"Running evaluation at epoch {epoch_idx + 1}...")
                
                # Evaluate using the specialized evaluation function
                # Randomize evaluation seed based on epoch
                eval_seed = args.seed + epoch_idx if args.eval_sample_seed is None else args.eval_sample_seed + epoch_idx
                eval_results = evaluate_local_flow_2d(
                    model=ema_model,
                    stats=stats,
                    out_path=f"{args.output_dir}/{exp_name}/eval_epoch_{epoch_idx + 1}.mp4",
                    num_envs=64,
                    max_steps=300,
                    pred_horizon=args.pred_horizon,
                    obs_horizon=args.obs_horizon,
                    action_horizon=args.action_horizon,
                    device=device,
                    eval_seed=eval_seed,  # Pass seed for randomization
                )
                
                # Check if this is the best model so far
                current_success_rate = eval_results['success_rate']
                if current_success_rate > best_success_rate:
                    best_success_rate = current_success_rate
                    best_epoch = epoch_idx + 1
                    
                    # Save best model
                    best_ckpt_path = os.path.join(ckpt_dir, f"best_ema_model.pt")
                    torch.save(ema_model.state_dict(), best_ckpt_path)
                    best_model_path = best_ckpt_path
                    print(f"New best model! Success rate: {best_success_rate:.4f} at epoch {best_epoch}")
                    print(f"Saved best model to: {best_ckpt_path}")
                
                # Prepare random number generator for all visualizations
                if args.eval_sample_seed is not None:
                    rng = np.random.RandomState(args.eval_sample_seed + epoch_idx)
                else:
                    rng = np.random.RandomState(args.seed + epoch_idx)
                
                # ====================================================================
                # TRAINING SET EVALUATIONS
                # ====================================================================
                print(f"\n{'='*60}")
                print(f"TRAINING SET EVALUATIONS (Epoch {epoch_idx + 1})")
                print(f"{'='*60}")
                
                train_action_results = None
                train_position_results = None
                train_traj_results = None
                
                train_sample_idx = rng.randint(0, len(dataset))
                print(f"  Sample index: {train_sample_idx}")
                
                # Evaluate action policy on training set
                print(f"  Evaluating action policy...")
                try:
                    train_action_results = visualize_action_predictions(
                        model=ema_model,
                        dataset=dataset,
                        stats=stats,
                        sample_idx=train_sample_idx,
                        out_path=f"{args.output_dir}/{exp_name}/eval_train_action_epoch_{epoch_idx + 1}.png",
                        device=device,
                    )
                    print(f"    ✓ Action error: {train_action_results.get('action_error', 0.0):.4f}")
                    print(f"    ✓ Saved to: {train_action_results['image_path']}")
                except Exception as e:
                    print(f"    ✗ Warning: Failed to evaluate action policy: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Evaluate position decoder (frame prediction) on training set
                if ema_model.position_decoder is not None:
                    print(f"  Evaluating position decoder (frame prediction)...")
                    try:
                        train_position_results = visualize_position_predictions(
                            model=ema_model,  # Pass full model so encoder is available
                            dataset=dataset,
                            stats=stats,
                            sample_idx=train_sample_idx,
                            out_path=f"{args.output_dir}/{exp_name}/eval_train_position_epoch_{epoch_idx + 1}.png",
                            device=device,
                        )
                        print(f"    ✓ Position error: {train_position_results.get('pred_error', 0.0):.4f}")
                        print(f"    ✓ Saved to: {train_position_results['image_path']}")
                    except Exception as e:
                        print(f"    ✗ Warning: Failed to evaluate position decoder: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  Skipping position decoder evaluation (position decoder not available)")
                
                # Visualize full trajectory on training set
                print(f"  Visualizing full trajectory...")
                try:
                    train_traj_results = visualize_training_trajectory(
                        model=ema_model,
                        dataset=dataset,
                        stats=stats,
                        sample_idx=train_sample_idx,
                        out_path=f"{args.output_dir}/{exp_name}/eval_train_trajectory_epoch_{epoch_idx + 1}.png",
                        device=device,
                    )
                    print(f"    ✓ Saved to: {train_traj_results['image_path']}")
                except Exception as e:
                    print(f"    ✗ Warning: Failed to visualize trajectory: {e}")
                    import traceback
                    traceback.print_exc()
                
                # ====================================================================
                # VALIDATION SET EVALUATIONS
                # ====================================================================
                val_action_results = None
                val_position_results = None
                val_traj_results = None
                
                if val_dataset is not None and len(val_dataset) > 0:
                    print(f"\n{'='*60}")
                    print(f"VALIDATION SET EVALUATIONS (Epoch {epoch_idx + 1})")
                    print(f"{'='*60}")
                    
                    val_sample_idx = rng.randint(0, len(val_dataset))
                    print(f"  Sample index: {val_sample_idx}")
                    
                    # Evaluate action policy on validation set
                    print(f"  Evaluating action policy...")
                    try:
                        val_action_results = visualize_action_predictions(
                            model=ema_model,
                            dataset=val_dataset,
                            stats=stats,
                            sample_idx=val_sample_idx,
                            out_path=f"{args.output_dir}/{exp_name}/eval_val_action_epoch_{epoch_idx + 1}.png",
                            device=device,
                        )
                        print(f"    ✓ Action error: {val_action_results.get('action_error', 0.0):.4f}")
                        print(f"    ✓ Saved to: {val_action_results['image_path']}")
                    except Exception as e:
                        print(f"    ✗ Warning: Failed to evaluate action policy: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Evaluate position decoder (frame prediction) on validation set
                    if ema_model.position_decoder is not None:
                        print(f"  Evaluating position decoder (frame prediction)...")
                        try:
                            val_position_results = visualize_position_predictions(
                                model=ema_model,  # Pass full model so encoder is available
                                dataset=val_dataset,
                                stats=stats,
                                sample_idx=val_sample_idx,
                                out_path=f"{args.output_dir}/{exp_name}/eval_val_position_epoch_{epoch_idx + 1}.png",
                                device=device,
                            )
                            print(f"    ✓ Position error: {val_position_results.get('pred_error', 0.0):.4f}")
                            print(f"    ✓ Saved to: {val_position_results['image_path']}")
                        except Exception as e:
                            print(f"    ✗ Warning: Failed to evaluate position decoder: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"  Skipping position decoder evaluation (position decoder not available)")
                    
                    # Visualize full trajectory on validation set
                    print(f"  Visualizing full trajectory...")
                    try:
                        val_traj_results = visualize_training_trajectory(
                            model=ema_model,
                            dataset=val_dataset,
                            stats=stats,
                            sample_idx=val_sample_idx,
                            out_path=f"{args.output_dir}/{exp_name}/eval_val_trajectory_epoch_{epoch_idx + 1}.png",
                            device=device,
                        )
                        print(f"    ✓ Saved to: {val_traj_results['image_path']}")
                    except Exception as e:
                        print(f"    ✗ Warning: Failed to visualize trajectory: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"\n{'='*60}")
                    print(f"VALIDATION SET EVALUATIONS (Epoch {epoch_idx + 1})")
                    print(f"{'='*60}")
                    print(f"  No validation dataset available, skipping validation evaluations.")
                
                print(f"\n{'='*60}\n")
                
                # ====================================================================
                # LOG TO WANDB
                # ====================================================================
                if wandb_available:
                    log_dict = {
                        "eval/epoch": epoch_idx + 1, 
                        "eval/Success Rate": eval_results['success_rate'],
                        "eval/mean_score": eval_results['mean_score'],
                        "eval/max_score": eval_results['max_score'],
                        "eval/mean_max_single_reward": eval_results['mean_max_single_reward'],
                        "eval/best_success_rate": best_success_rate,
                        "eval/best_epoch": best_epoch,
                    }
                    
                    # Log training set evaluations
                    if train_action_results is not None:
                        log_dict["eval_train/action_error"] = train_action_results.get('action_error', 0.0)
                        if train_action_results.get('image_path') and os.path.exists(train_action_results['image_path']):
                            log_dict["eval_train/action_prediction"] = wandb.Image(train_action_results['image_path'])
                    
                    if train_position_results is not None:
                        log_dict["eval_train/position_error"] = train_position_results.get('pred_error', 0.0)
                        if train_position_results.get('image_path') and os.path.exists(train_position_results['image_path']):
                            log_dict["eval_train/position_prediction"] = wandb.Image(train_position_results['image_path'])
                    
                    if train_traj_results is not None and train_traj_results.get('image_path') is not None:
                        image_path = train_traj_results['image_path']
                        if os.path.exists(image_path):
                            log_dict["eval_train/trajectory"] = wandb.Image(image_path)
                    
                    # Log validation set evaluations
                    if val_action_results is not None:
                        log_dict["eval_val/action_error"] = val_action_results.get('action_error', 0.0)
                        if val_action_results.get('image_path') and os.path.exists(val_action_results['image_path']):
                            log_dict["eval_val/action_prediction"] = wandb.Image(val_action_results['image_path'])
                    
                    if val_position_results is not None:
                        log_dict["eval_val/position_error"] = val_position_results.get('pred_error', 0.0)
                        if val_position_results.get('image_path') and os.path.exists(val_position_results['image_path']):
                            log_dict["eval_val/position_prediction"] = wandb.Image(val_position_results['image_path'])
                    
                    if val_traj_results is not None and val_traj_results.get('image_path') is not None:
                        image_path = val_traj_results['image_path']
                        if os.path.exists(image_path):
                            log_dict["eval_val/trajectory"] = wandb.Image(image_path)
                    
                    # Log environment evaluation video if available
                    if eval_results.get('video_path') is not None:
                        video_path = eval_results['video_path']
                        if os.path.exists(video_path):
                            log_dict["eval/environment_video"] = wandb.Video(video_path, fps=30, format="mp4")
                    
                    wandb.log(log_dict)

    # Copy EMA weights to a fresh model and save checkpoint
    ema_model = LocalFlowPolicy2D(
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
        position_decoder_particles_aggregation=args.position_decoder_particles_aggregation,
        position_loss_coeff=args.position_loss_coeff,
        share_noise=args.share_noise,
        shared_noise_base=args.shared_noise_base,
        use_gt_reference_for_local_policy=args.use_gt_reference_for_local_policy,
        use_film_conditioning=args.use_film_conditioning,
        film_hidden_dim=args.film_hidden_dim,
        film_predict_scale=args.film_predict_scale,
        # input_noise_std=args.input_noise_std,
        disable_reference_conditioning=args.disable_reference_conditioning,
    ).to(device)
    
    ema.copy_to(ema_model.parameters())
    
    # save final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"ema_model.pt")
    torch.save(ema_model.state_dict(), ckpt_path)
    print(f"Saved final checkpoint: {ckpt_path}")
    
    # Print summary of best model
    if best_model_path is not None:
        print(f"\n{'='*50}")
        print("Best Model Summary:")
        print(f"{'='*50}")
        print(f"Best Success Rate: {best_success_rate:.4f}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Model Path: {best_model_path}")
        print(f"{'='*50}")
    else:
        print("Note: No evaluations were performed, so no best model was saved.")

    if wandb_available:
        try:
            wandb.save(ckpt_path)
            wandb.finish()
        except Exception:
            pass

