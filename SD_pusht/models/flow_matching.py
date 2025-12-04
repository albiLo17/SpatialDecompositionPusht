"""Flow Matching model class for action prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditional_unet import ConditionalUnet1D
from .conditional_flow_matcher import ConditionalFlowMatcher


class FlowMatching(nn.Module):
    """
    Flow Matching model for action prediction.
    
    This class encapsulates the flow matching process including:
    - Velocity prediction network (ConditionalUnet1D)
    - Conditional Flow Matcher
    - Training loss computation
    - Action sampling/inference
    """
    
    def __init__(
        self,
        act_dim: int,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        obs_dim: int,
        sigma: float = 0.0,
        fm_timesteps: int = 100,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = None,
        n_groups: int = 8
    ):
        """
        Args:
            act_dim: Action dimension
            obs_horizon: Observation horizon
            act_horizon: Action horizon
            pred_horizon: Prediction horizon
            obs_dim: Observation dimension
            sigma: Noise scale for flow matching paths
            fm_timesteps: Number of timesteps for flow matching inference
            diffusion_step_embed_dim: Dimension of timestep embedding
            down_dims: Channel dimensions for U-Net downsampling
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.fm_timesteps = fm_timesteps
        
        if down_dims is None:
            down_dims = [256, 512, 1024]
        
        # Velocity prediction network (reuses ConditionalUnet1D architecture)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=obs_horizon * obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            n_groups=n_groups,
        )
        
        # Conditional Flow Matcher
        self.FM = ConditionalFlowMatcher(sigma=sigma)
    
    def compute_loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow matching training loss.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            action_seq: Action sequence, shape (B, pred_horizon, act_dim)
            
        Returns:
            Loss value
        """
        # Sample noise x0
        x0 = torch.randn(action_seq.shape, device=obs_seq.device)
        
        # Sample location along path and get conditional flow
        timestep, xt, ut = self.FM.sample_location_and_conditional_flow(x0, action_seq)
        
        # Encode observation features
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        
        # Predict velocity
        vt = self.noise_pred_net(xt, timestep, global_cond=obs_cond)
        
        # Compute MSE loss between predicted and target velocity
        return torch.mean((vt - ut) ** 2)  # or F.mse_loss(vt, ut)
    
    def get_action(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Sample actions from the flow matching model.
        
        Args:
            obs_seq: Observation sequence, shape (B, obs_horizon, obs_dim)
            
        Returns:
            Actions, shape (B, act_horizon, act_dim)
        """
        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
            
            # Initialize action from Gaussian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq.device
            )
            
            # Integrate the flow over timesteps
            for i in range(self.fm_timesteps):
                timestep = torch.tensor(
                    [i / self.fm_timesteps], device=obs_seq.device
                ).repeat(B,)
                
                if i == 0:
                    vt = self.noise_pred_net(
                        noisy_action_seq, timestep, global_cond=obs_cond)
                    denoised_action_seq = (vt * 1 / self.fm_timesteps + noisy_action_seq)
                else:
                    vt = self.noise_pred_net(
                        denoised_action_seq, timestep, global_cond=obs_cond)
                    denoised_action_seq = (vt * 1 / self.fm_timesteps + denoised_action_seq)
        
        # Only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return denoised_action_seq[:, start:end]  # (B, act_horizon, act_dim)

