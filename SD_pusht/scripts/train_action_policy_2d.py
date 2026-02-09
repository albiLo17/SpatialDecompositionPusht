#!/usr/bin/env python3
"""Training script for Action Policy (LocalFlowPolicy2D) on PushT with ground truth poses.

This script trains only the action policy component, using ground truth reference
positions (oracle mode). The position decoder is not trained.

python SD_pusht/scripts/train_action_policy_2d.py \
    --dataset-path datasets/pusht_cchi_v7_replay.zarr.zip \
    --max-demos 200 \
    --batch-size 256 \
    --epochs 1000 \
    --wandb
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
from SD_pusht.utils.evaluation import visualize_action_predictions
from SD_pusht.utils.normalization import unnormalize_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train Action Policy with GT poses on PushT")
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
    parser.add_argument("--wandb-project", type=str, default="sd-pusht-action-policy",
                       help="WandB project name")
    parser.add_argument("--eval-every", type=int, default=10,
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
    # Dataset segmentation arguments
    parser.add_argument("--contact-threshold", type=float, default=0.1,
                       help="Threshold for detecting block movement (contact)")
    parser.add_argument("--min-segment-length", type=int, default=5,
                       help="Minimum length for a valid segment")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Fraction of episodes to use for validation")
    parser.add_argument("--val-min-episodes", type=int, default=10,
                       help="Minimum number of episodes for validation set")
    parser.add_argument("--eval-sample-seed", type=int, default=None,
                       help="Random seed for selecting evaluation sample (None = use epoch number)")
    parser.add_argument("--transform-obs-to-local-frame", action="store_true",
                       help="Transform observations to local frame (relative to reference position) before using them")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment name
    local_frame_suffix = "-localobs" if args.transform_obs_to_local_frame else ""
    exp_name = f"sd-pusht-action-policy-demos-{args.max_demos}-seed{args.seed}{local_frame_suffix}"
    print(f"Experiment: {exp_name}")
    print(f"Using model: LocalFlowPolicy2D (action only, with GT poses)")
    if args.transform_obs_to_local_frame:
        print("Using local frame observations (transformed relative to reference position)")
    
    # Create experiment directory and save config
    exp_dir = f"{args.output_dir}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    
    # Convert args to dict and save as JSON
    config_dict = vars(args).copy()
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
    
    # Create training dataset
    dataset = PushTSegmentedDatasetSimple(
        dataset_path=dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        max_demos=args.max_demos,
        use_contact_segmentation=True,
        contact_threshold=args.contact_threshold,
        min_segment_length=args.min_segment_length,
    )
    
    # Get demo indices used by training dataset
    train_demo_indices = getattr(dataset, 'demo_indices', None)
    if train_demo_indices is None:
        train_demo_indices = list(range(args.max_demos if args.max_demos is not None else total_episodes))
    
    print(f"Training dataset uses {len(train_demo_indices)} episodes")
    
    # Calculate validation demo indices
    val_dataset = None
    if args.val_split > 0:
        all_demo_indices = set(range(total_episodes))
        train_demo_indices_set = set(train_demo_indices)
        remaining_demo_indices = sorted(list(all_demo_indices - train_demo_indices_set))
        
        if len(remaining_demo_indices) >= args.val_min_episodes:
            val_demo_indices = remaining_demo_indices
            print(f"Using {len(val_demo_indices)} remaining episodes for validation")
        else:
            val_size = max(args.val_min_episodes, int(len(train_demo_indices) * args.val_split))
            rng = np.random.RandomState(args.seed)
            val_demo_indices = sorted(rng.choice(train_demo_indices, size=val_size, replace=False).tolist())
            print(f"Randomly selected {len(val_demo_indices)} episodes from training for validation")
        
        try:
            val_dataset = PushTSegmentedDatasetSimple(
                dataset_path=dataset_path,
                pred_horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                action_horizon=args.action_horizon,
                demo_indices=val_demo_indices,
                use_contact_segmentation=True,
                contact_threshold=args.contact_threshold,
                min_segment_length=args.min_segment_length,
            )
            print(f"Created validation dataset with {len(val_dataset)} samples")
        except Exception as e:
            print(f"Warning: Could not create validation dataset: {e}")
            val_dataset = None

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

    # create LocalFlowPolicy2D model (without position decoder, using GT poses)
    agent = LocalFlowPolicy2D(
        act_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        act_horizon=args.action_horizon,
        pred_horizon=args.pred_horizon,
        obs_dim=args.obs_dim,
        sigma=args.sigma,
        fm_timesteps=args.fm_timesteps,
        use_position_decoder=False,  # Don't use position decoder
        use_gt_reference_for_local_policy=True,  # Use GT poses (oracle mode)
        transform_obs_to_local_frame=args.transform_obs_to_local_frame,  # Transform obs to local frame
    ).to(device)
    
    if args.transform_obs_to_local_frame:
        print("Using local frame observations (transformed relative to reference position)")

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

    # Create checkpoint directory
    ckpt_dir = f"{args.output_dir}/{exp_name}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            epoch_action_loss = list()
            
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    reference_pos = nbatch['reference_pos'].to(device)

                    # Compute loss using the model's compute_loss method
                    # Since use_gt_reference_for_local_policy=True, it will use GT poses
                    # Pass stats for proper normalization/unnormalization
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
                    
                    tepoch.set_postfix(
                        loss=loss_cpu,
                        action_loss=action_loss,
                    )

            # Log at epoch level
            mean_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0.0
            mean_action_loss = np.mean(epoch_action_loss) if epoch_action_loss else 0.0
            
            tglobal.set_postfix(
                loss=mean_epoch_loss,
                action_loss=mean_action_loss,
            )
            
            if wandb_available:
                log_dict = {
                    "train/epoch_loss": mean_epoch_loss,
                    "train/action_loss": mean_action_loss,
                    "train/lr": lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else None,
                    "train/epoch": epoch_idx
                }
                wandb.log(log_dict)
            
            # Evaluation step - visualize on both training and validation datasets
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
                    use_position_decoder=False,
                    use_gt_reference_for_local_policy=True,
                    transform_obs_to_local_frame=args.transform_obs_to_local_frame,
                ).to(device)
                
                ema.copy_to(ema_model.parameters())
                ema_model.eval()
                print(f"Running evaluation at epoch {epoch_idx + 1}...")
                
                # Prepare random number generator
                if args.eval_sample_seed is not None:
                    rng = np.random.RandomState(args.eval_sample_seed + epoch_idx)
                else:
                    rng = np.random.RandomState(args.seed + epoch_idx)
                
                eval_log_dict = {"eval/epoch": epoch_idx + 1}
                
                # Evaluate on training dataset
                try:
                    train_sample_idx = rng.randint(0, len(dataset))
                    print(f"  Evaluating on training dataset (sample {train_sample_idx})...")
                    
                    train_vis_results = visualize_action_predictions(
                        model=ema_model,
                        dataset=dataset,
                        stats=stats,
                        sample_idx=train_sample_idx,
                        out_path=f"{args.output_dir}/{exp_name}/action_pred_train_epoch_{epoch_idx + 1}.png",
                        device=device,
                    )
                    print(f"  Saved training visualization to: {train_vis_results['image_path']}")
                    
                    eval_log_dict["eval/train_action_error"] = train_vis_results.get('action_error', 0.0)
                    if wandb_available:
                        eval_log_dict["eval/train_action_prediction"] = wandb.Image(train_vis_results['image_path'])
                except Exception as e:
                    print(f"  Warning: Failed to visualize on training dataset: {e}")
                
                # Evaluate on validation dataset (if available)
                if val_dataset is not None and len(val_dataset) > 0:
                    try:
                        val_sample_idx = rng.randint(0, len(val_dataset))
                        print(f"  Evaluating on validation dataset (sample {val_sample_idx})...")
                        
                        val_vis_results = visualize_action_predictions(
                            model=ema_model,
                            dataset=val_dataset,
                            stats=stats,
                            sample_idx=val_sample_idx,
                            out_path=f"{args.output_dir}/{exp_name}/action_pred_val_epoch_{epoch_idx + 1}.png",
                            device=device,
                        )
                        print(f"  Saved validation visualization to: {val_vis_results['image_path']}")
                        
                        eval_log_dict["eval/val_action_error"] = val_vis_results.get('action_error', 0.0)
                        if wandb_available:
                            eval_log_dict["eval/val_action_prediction"] = wandb.Image(val_vis_results['image_path'])
                    except Exception as e:
                        print(f"  Warning: Failed to visualize on validation dataset: {e}")
                
                # Log to wandb
                if wandb_available and eval_log_dict:
                    wandb.log(eval_log_dict)

    # Copy EMA weights and save checkpoint
    ema_model = LocalFlowPolicy2D(
        act_dim=args.action_dim,
        obs_horizon=args.obs_horizon,
        act_horizon=args.action_horizon,
        pred_horizon=args.pred_horizon,
        obs_dim=args.obs_dim,
        sigma=args.sigma,
        fm_timesteps=args.fm_timesteps,
        use_position_decoder=False,
        use_gt_reference_for_local_policy=True,
        transform_obs_to_local_frame=args.transform_obs_to_local_frame,
    ).to(device)
    
    ema.copy_to(ema_model.parameters())
    
    # save final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"action_policy.pt")
    torch.save(ema_model.state_dict(), ckpt_path)
    print(f"Saved final checkpoint: {ckpt_path}")

    if wandb_available:
        try:
            wandb.save(ckpt_path)
            wandb.finish()
        except Exception:
            pass

