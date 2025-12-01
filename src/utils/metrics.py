import glob # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from tqdm import tqdm
import os

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_batch(batch):
    """Normalize each image in batch to [0, 1] independently."""
    rescaled = []
    for i in range(batch.shape[0]):
        img = batch[i]
        img_rescaled = (img - img.min()) / (img.max() - img.min() + 1e-8)
        rescaled.append(img_rescaled)
    return torch.stack(rescaled)

def compute_topographic_rmse(gt, pred):
    """Compute RMSE between gradients of GT and prediction (topographic RMSE)."""
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    dx_rmse = F.mse_loss(pred_dx, gt_dx).sqrt()
    dy_rmse = F.mse_loss(pred_dy, gt_dy).sqrt()
    return (dx_rmse + dy_rmse) / 2

def masked_mse_loss(pred, target, mask):
    """Calculate MSE loss with mask weighting."""
    return ((pred - target) ** 2 * mask.unsqueeze(1)).sum() / mask.sum()

def masked_mae_loss(pred, target, mask):
    """Calculate MAE loss with mask weighting."""
    return (torch.abs(pred - target) * mask.unsqueeze(1)).sum() / mask.sum()

def masked_hybrid_mse_loss(pred, target, mask, alpha=0.1):
    """
    Calculate hybrid MSE + gradient loss.
    Returns: total_loss, pixel_loss, gradient_loss
    """
    # Pixel-wise MSE loss
    pixel_loss = masked_mse_loss(pred, target, mask)

    # Gradient loss (MSE of the gradients)
    gt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    grad_loss_x = F.mse_loss(pred_dx, gt_dx)

    gt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    grad_loss_y = F.mse_loss(pred_dy, gt_dy)
    
    gradient_loss = (grad_loss_x + grad_loss_y) / 2
    
    total_loss = pixel_loss + alpha * gradient_loss
    return total_loss, pixel_loss, gradient_loss

def masked_hybrid_mae_loss(pred, target, mask, alpha=0.1):
    """
    Calculate hybrid MAE + gradient loss.
    Returns: total_loss, pixel_loss, gradient_loss
    """
    # Pixel-wise MAE loss
    pixel_loss = masked_mae_loss(pred, target, mask)

    # Gradient loss (L1 loss of the gradients)
    gt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    grad_loss_x = F.l1_loss(pred_dx, gt_dx)

    gt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    grad_loss_y = F.l1_loss(pred_dy, gt_dy)
    
    gradient_loss = (grad_loss_x + grad_loss_y) / 2

    total_loss = pixel_loss + alpha * gradient_loss
    return total_loss, pixel_loss, gradient_loss