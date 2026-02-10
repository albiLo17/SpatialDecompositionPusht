"""Model definitions for SD_pusht."""

from .conditional_unet import ConditionalUnet1D
from .conditional_flow_matcher import ConditionalFlowMatcher
from .diffusion import Diffusion
from .flow_matching import FlowMatching
from .local_flow_policy import LocalFlowPolicy, PoseFlowDecoder
from .local_flow_policy_2d import LocalFlowPolicy3D, Position3DFlowDecoder, PositionMLP, DirectPositionMLP, FiLMReEncoder3D

# Backward compatibility aliases (for existing code that uses 2D names)
LocalFlowPolicy2D = LocalFlowPolicy3D
Position2DFlowDecoder = Position3DFlowDecoder

__all__ = [
    "ConditionalUnet1D",
    "ConditionalFlowMatcher",
    "Diffusion",
    "FlowMatching",
    "LocalFlowPolicy",
    "PoseFlowDecoder",
    "LocalFlowPolicy3D",
    "Position3DFlowDecoder",
    "PositionMLP",
    "DirectPositionMLP",
    "FiLMReEncoder3D",
    # Backward compatibility
    "LocalFlowPolicy2D",
    "Position2DFlowDecoder",
]

