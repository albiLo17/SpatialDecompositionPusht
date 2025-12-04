"""Conditional Flow Matcher for flow matching training."""

import torch
import torch.nn as nn


class ConditionalFlowMatcher(nn.Module):
    """
    Conditional Flow Matcher for training flow matching models.
    
    Implements the conditional flow matching objective where we learn to predict
    the flow field that transports noise to data.
    """
    
    def __init__(self, sigma: float = 0.0):
        """
        Args:
            sigma: Noise scale parameter. If 0, uses straight line paths.
                   If > 0, adds Gaussian noise to the path.
        """
        super().__init__()
        self.sigma = sigma
    
    def sample_location_and_conditional_flow(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random point along the path from x0 to x1 and compute the conditional flow.
        
        Args:
            x0: Starting point (noise), shape (B, ...)
            x1: Target point (data), shape (B, ...)
            
        Returns:
            timestep: Random timestep t in [0, 1], shape (B,)
            xt: Point along the path at timestep t, shape (B, ...)
            ut: Conditional flow (target velocity), shape (B, ...)
        """
        B = x0.shape[0]
        device = x0.device
        
        # Sample random timestep uniformly from [0, 1]
        t = torch.rand(B, device=device)
        
        # Compute point along the path: xt = (1-t) * x0 + t * x1
        # Add noise if sigma > 0
        if self.sigma > 0:
            # Add Gaussian noise scaled by sqrt(t * (1-t))
            noise = torch.randn_like(x0)
            noise_scale = self.sigma * torch.sqrt(t * (1 - t))
            # Reshape noise_scale to broadcast correctly
            while len(noise_scale.shape) < len(x0.shape):
                noise_scale = noise_scale.unsqueeze(-1)
            t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))
            xt = (1 - t_expanded) * x0 + t_expanded * x1 + noise_scale * noise
        else:
            # Straight line path without noise
            t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))
            xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Conditional flow is the direction from x0 to x1: ut = x1 - x0
        ut = x1 - x0
        
        return t, xt, ut

