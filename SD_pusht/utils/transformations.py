"""Transformation utilities for pose and frame operations.

This module provides utilities for working with transformation matrices,
rotation representations, and pose transformations. It includes fallback
implementations when optional dependencies (casino, roma) are not available.
"""

from typing import Optional
import torch

# Try to import optional dependencies for transformations
try:
    import casino
    import roma
    HAS_CASINO = True
except ImportError:
    HAS_CASINO = False


def to_transformation_matrix_th(t: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Create 4x4 transformation matrix from translation and rotation.
    
    Args:
        t: Translation vector, shape (..., 3)
        R: Rotation matrix, shape (..., 3, 3)
    
    Returns:
        Transformation matrix, shape (..., 4, 4)
    """
    assert t.shape[-1] == 3, f"Expected last dimension of t to be 3, got: {t.shape}"
    assert R.shape[-2:] == (3, 3), f"Expected last two dimensions of R to be (3, 3), got: {R.shape}"
    
    batch_shape = t.shape[:-1]
    device = t.device
    
    # Create identity matrices
    T = torch.eye(4, device=device).expand(*batch_shape, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    
    return T


def rotvec_to_rotmat(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vector (axis-angle) to rotation matrix.
    
    Args:
        rotvec: Rotation vector, shape (..., 3)
    
    Returns:
        Rotation matrix, shape (..., 3, 3)
    """
    if HAS_CASINO:
        return roma.rotvec_to_rotmat(rotvec)
    
    # Manual implementation using Rodrigues' formula
    angle = torch.norm(rotvec, dim=-1, keepdim=True)
    angle = angle.clamp(min=1e-8)  # Avoid division by zero
    
    axis = rotvec / angle
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    # Rodrigues' rotation formula
    K = torch.zeros(*rotvec.shape[:-1], 3, 3, device=rotvec.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    R = (torch.eye(3, device=rotvec.device).expand_as(K) * cos_a.unsqueeze(-1) +
         (1 - cos_a).unsqueeze(-1) * torch.einsum('...i,...j->...ij', axis, axis) +
         sin_a.unsqueeze(-1) * K)
    
    return R


def rotmat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to rotation vector (axis-angle).
    
    Args:
        R: Rotation matrix, shape (..., 3, 3)
    
    Returns:
        Rotation vector, shape (..., 3)
    """
    if HAS_CASINO:
        return roma.rotmat_to_rotvec(R)
    
    # Manual implementation
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # Handle small angles
    small_angle_mask = angle < 1e-6
    angle = angle.unsqueeze(-1)
    
    # Compute axis
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis / axis_norm
    
    # For small angles, use alternative formula
    rotvec = angle * axis
    rotvec[small_angle_mask] = 0.5 * axis[small_angle_mask]
    
    return rotvec


def transform_flat_pose_vector_th(
    T: torch.Tensor,
    v: torch.Tensor,
    pre_multiply: bool = True
) -> torch.Tensor:
    """Transform a flat pose vector (xyz + rotvec) by a transformation matrix.
    
    Args:
        T: Transformation matrix, shape (..., 4, 4)
        v: Flat pose vector (xyz, rotvec), shape (..., 6)
        pre_multiply: If True, apply T before v (T @ V), else after (V @ T)
    
    Returns:
        Transformed flat pose vector, shape (..., 6)
    """
    assert v.shape[-1] == 6, f"Expected last dimension of v to be 6, got: {v.shape}"
    assert T.shape[-2:] == (4, 4), f"Expected last two dimensions of T to be (4, 4), got: {T.shape}"
    
    v_xyz = v[..., :3]
    v_rotvec = v[..., 3:6]
    v_rotmat = rotvec_to_rotmat(v_rotvec)
    
    V = to_transformation_matrix_th(t=v_xyz, R=v_rotmat)
    
    if pre_multiply:
        transformed_V = torch.matmul(T, V)
    else:
        transformed_V = torch.matmul(V, T)
    
    transformed_V_xyz = transformed_V[..., :3, 3]
    transformed_V_rotmat = transformed_V[..., :3, :3]
    transformed_V_rotvec = rotmat_to_rotvec(transformed_V_rotmat)
    
    transformed_v = torch.cat([transformed_V_xyz, transformed_V_rotvec], dim=-1)
    return transformed_v


def special_gramschmidt(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Apply Gram-Schmidt orthonormalization to two vectors to form rotation matrix.
    
    Args:
        v1: First vector, shape (..., 3)
        v2: Second vector, shape (..., 3)
    
    Returns:
        Rotation matrix, shape (..., 3, 3)
    """
    if HAS_CASINO:
        return roma.special_gramschmidt(torch.stack([v1, v2], dim=-1))
    
    # Manual Gram-Schmidt
    u1 = v1 / torch.norm(v1, dim=-1, keepdim=True).clamp(min=1e-8)
    v2_proj = (v2 * u1).sum(dim=-1, keepdim=True) * u1
    u2 = v2 - v2_proj
    u2 = u2 / torch.norm(u2, dim=-1, keepdim=True).clamp(min=1e-8)
    u3 = torch.cross(u1, u2, dim=-1)
    
    R = torch.stack([u1, u2, u3], dim=-1)
    return R

