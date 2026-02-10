#!/usr/bin/env python3
"""Test script to evaluate a random policy on SoftGym environment.

This script:
1. Creates a random policy (generates random actions)
2. Runs it in the SoftGym environment
3. Evaluates performance metrics

Usage:
    python SpatialDecompositionPusht/SD_pusht/scripts/test_eval_softgym.py \
        --env-name RopeFlatten \
        --num-envs 4 \
        --max-steps 100
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from SD_pusht.utils.evaluation_softgym import (
    build_softgym_env,
    split_state_from_obs,
    evaluate_local_flow_3d_softgym
)
from SD_pusht.models import LocalFlowPolicy3D


class RandomPolicy:
    """Random policy that generates random actions."""
    
    def __init__(self, action_dim: int, num_pickers: int = 2):
        """Initialize random policy.
        
        Args:
            action_dim: Action dimension (should be 4 * num_pickers)
            num_pickers: Number of pickers/grippers
        """
        self.action_dim = action_dim
        self.num_pickers = num_pickers
    
    def get_action(self, obs_seq, **kwargs):
        """Generate random action.
        
        Args:
            obs_seq: Observation sequence (ignored for random policy)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Dictionary with 'actions' key containing random actions
        """
        batch_size = obs_seq.shape[0] if isinstance(obs_seq, torch.Tensor) else 1
        action_horizon = 8  # Default action horizon
        
        # Generate random actions
        # Actions are [dx1, dy1, dz1, pick1, dx2, dy2, dz2, pick2, ...]
        # Translation: small random values in [-0.01, 0.01]
        # Pick flag: random 0 or 1
        actions = np.random.uniform(
            low=[-0.01, -0.01, -0.01, 0.0] * self.num_pickers,
            high=[0.01, 0.01, 0.01, 1.0] * self.num_pickers,
            size=(batch_size, action_horizon, self.action_dim)
        ).astype(np.float32)
        
        # Convert to torch tensor
        if isinstance(obs_seq, torch.Tensor):
            actions = torch.from_numpy(actions).to(obs_seq.device)
        
        return {
            "actions": actions,
            "reference_pos": None,  # Random policy doesn't predict reference
        }


def test_random_policy(
    env_name: str = "RopeFlatten",
    num_envs: int = 4,
    max_steps: int = 100,
    num_pickers: int = 2,
    env_kwargs: dict = None,
    seed: int = 42,
):
    """Test a random policy on SoftGym environment.
    
    Args:
        env_name: SoftGym environment name
        num_envs: Number of parallel environments
        max_steps: Maximum steps per episode
        num_pickers: Number of pickers/grippers
        env_kwargs: Optional environment kwargs
        seed: Random seed
    
    Returns:
        Dictionary with evaluation results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("="*60)
    print("Testing Random Policy on SoftGym Environment")
    print("="*60)
    print(f"Environment: {env_name}")
    print(f"Number of environments: {num_envs}")
    print(f"Max steps: {max_steps}")
    print(f"Number of pickers: {num_pickers}")
    print()
    

    obs_dim = 36
    
    # Fallback: assume standard action dim
    action_dim = 4 * num_pickers
    
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print()
    
    # Create random policy
    policy = RandomPolicy(action_dim=action_dim, num_pickers=num_pickers)
    
    # Create dummy stats (for normalization - random policy doesn't need real stats)
    dummy_stats = {
        "obs": {
            "min": np.zeros(obs_dim, dtype=np.float32),
            "max": np.ones(obs_dim, dtype=np.float32),
        },
        "action": {
            "min": np.array([-0.1, -0.1, -0.1, 0.0] * num_pickers, dtype=np.float32),
            "max": np.array([0.1, 0.1, 0.1, 1.0] * num_pickers, dtype=np.float32),
        },
    }
    
    # Wrap policy in a model-like interface for evaluation
    class RandomPolicyWrapper:
        """Wrapper to make random policy compatible with evaluation function."""
        
        def __init__(self, policy, device):
            self.policy = policy
            self.device = device
            self.eval = lambda: None  # Dummy eval method
        
        def get_action(self, obs_seq, **kwargs):
            """Get action from random policy."""
            # Convert torch to numpy if needed
            if isinstance(obs_seq, torch.Tensor):
                obs_seq_np = obs_seq.cpu().numpy()
            else:
                obs_seq_np = obs_seq
            
            action_dict = self.policy.get_action(obs_seq_np, **kwargs)
            
            # Convert back to torch if input was torch
            if isinstance(obs_seq, torch.Tensor):
                if isinstance(action_dict["actions"], np.ndarray):
                    action_dict["actions"] = torch.from_numpy(action_dict["actions"]).to(self.device)
            
            return action_dict
        
        def parameters(self):
            """Dummy parameters for device detection."""
            return [torch.zeros(1)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RandomPolicyWrapper(policy, device)
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluate_local_flow_3d_softgym(
        model=model,
        stats=dummy_stats,
        env_name=env_name,
        env_kwargs=env_kwargs,
        out_path="test_random_policy.mp4",
        num_envs=num_envs,
        max_steps=max_steps,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        device=device,
        save_video=True,
        eval_seed=seed,
        num_pickers=num_pickers,
    )
    
    print()
    print("="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Mean Score: {results['mean_score']:.4f}")
    print(f"Max Score: {results['max_score']:.4f}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Steps Done: {results['steps_done']}")
    if results.get('video_path'):
        print(f"Video saved to: {results['video_path']}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test random policy on SoftGym environment")
    parser.add_argument("--env-name", type=str, default="RopeFlatten",
                       help="SoftGym environment name")
    parser.add_argument("--num-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--num-pickers", type=int, default=2,
                       help="Number of pickers/grippers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--env-horizon", type=int, default=None,
                       help="Override environment horizon")
    
    args = parser.parse_args()
    
    env_kwargs = {}
    if args.env_horizon is not None:
        env_kwargs["horizon"] = args.env_horizon
    
    test_random_policy(
        env_name=args.env_name,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        num_pickers=args.num_pickers,
        env_kwargs=env_kwargs if env_kwargs else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
