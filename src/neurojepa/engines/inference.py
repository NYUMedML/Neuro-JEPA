import torch
import sys
import logging
import os
import pickle
from typing import Any, Dict, Optional

import torch.distributed as dist
import torch.nn.functional as F

from neurojepa.models.utils.moe import MoE
from neurojepa.utils.misc import MetricLogger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
    

def get_foreground_mask(volumes, patch_size=(16, 16, 4), min_foreground_fraction=0.1):
    """
    Compute a per-patch binary foreground map from a batch of 3D volumes.
    Uses dynamic per-sample thresholding: threshold = 0.1 * (P98 - P2) + P2
    """
    # Assuming volumes is (B, C, D, H, W) based on valid Conv3d input
    
    # Check if patch_size matches D, H, W
    B, C, D, H, W = volumes.shape
    pD, pH, pW = patch_size # Typically (16, 16, 4) means D=16, H=16, W=4 ? 
    
    # Flatten spatial dims to get per-sample stats
    # Use max projection across channels (matches masking.py)
    vol_max = volumes.abs().amax(dim=1, keepdim=True) # (B, 1, D, H, W)
    vol_flat = vol_max.reshape(B, -1)
    
    p2 = torch.quantile(vol_flat.float(), 0.02, dim=1)
    p98 = torch.quantile(vol_flat.float(), 0.98, dim=1)
    dyn_thresh = 0.1 * (p98 - p2) + p2
    
    # Fallback default threshold
    threshold = 0.0
    dyn_thresh = torch.where(p98 > p2, dyn_thresh, torch.full_like(dyn_thresh, threshold))
    
    dyn_thresh = dyn_thresh.view(B, 1, 1, 1, 1)

    # Collapse channel dim
    vol = vol_max # Already max projected
    
    # Unfold into patches
    # Assuming patch_size corresponds to (pD, pH, pW) of the input tensor (D, H, W)
    
    # Check divisibility
    # pad if needed (as per original get_foreground_mask, though masking.py asserts divisibility)
    pad_D = (pD - D % pD) % pD
    pad_H = (pH - H % pH) % pH
    pad_W = (pW - W % pW) % pW
    if pad_D > 0 or pad_H > 0 or pad_W > 0:
        vol = F.pad(vol, (0, pad_W, 0, pad_H, 0, pad_D))
        # D, H, W update not needed for unfold parameters, but for shape
        
    u_vol = vol.unfold(2, pD, pD).unfold(3, pH, pH).unfold(4, pW, pW)
    # Shape: (B, 1, nD, nH, nW, pD, pH, pW)
    
    # Calculate fraction of pixels > threshold per patch
    dt = dyn_thresh.view(B, 1, 1, 1, 1, 1, 1, 1)
    fg_frac = (u_vol > dt).float().mean(dim=(-3, -2, -1))
    
    # (B, 1, nD, nH, nW) -> (B, N)
    # Flatten matching PatchEmbed3D: flatten(2) merges D, H, W.
    return (fg_frac >= min_foreground_fraction).flatten(1) # Remove channel dim and flatten spatial


def infer_one_epoch(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:
    """Train the model for one epoch for multi-class classification."""
    metric_logger = MetricLogger(delimiter="  ", logger=logger)
    backbone = model['backbone']
    backbone.eval()

    use_amp = cfg.optimization.mixed_precision

    backbone_noddp = backbone.module
    num_blocks = len(backbone_noddp.blocks)
    num_experts = cfg.model.moe_params.n_routed_experts
    epoch_expert_counts_sum = {i: torch.zeros(num_experts, device=device) for i in range(num_blocks)}
    epoch_expert_counts_fg = {i: torch.zeros(num_experts, device=device) for i in range(num_blocks)}
    epoch_expert_counts_bg = {i: torch.zeros(num_experts, device=device) for i in range(num_blocks)}
    total_tokens_epoch = 0

    # Per-token-position routing: maps block_idx -> sequential MoE layer index
    moe_block_indices = [i for i, b in enumerate(backbone_noddp.blocks) if isinstance(b.mlp, MoE)]
    num_moe_layers = len(moe_block_indices)
    moe_block_to_layer = {bi: li for li, bi in enumerate(moe_block_indices)}
    token_routing_counts = None  # Lazily initialized to [num_moe_layers, num_tokens, num_experts]
    
    for idx, batch_data in enumerate(loader):
        print(f"Inference step {idx+1}/{len(loader)}", end='\r')
        image, target = batch_data
        image = image.to(device, non_blocking=True)
        
        # Calculate foreground mask
        patch_size = backbone_noddp.patch_size if hasattr(backbone_noddp, 'patch_size') else (16, 16, 4)
        fg_mask = get_foreground_mask(image, patch_size).to(device) # (B, N)
        fg_mask_flat = fg_mask.view(-1) # (B*N)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
            feats = backbone(image)
        
        torch.cuda.synchronize()

        # get expert bias update for MoE in each layer
        for block_idx, block in enumerate(backbone_noddp.blocks):
            # check if block has MoE layer
            if not isinstance(block.mlp, MoE):
                continue
            moe_layer = block.mlp
            expert_counts = moe_layer.counts.clone()
            
            # Use stored indices to separate counts
            if hasattr(moe_layer, 'last_indices') and moe_layer.last_indices is not None:
                indices = moe_layer.last_indices # (B*N, k)

                # Accumulate per-token-position routing counts -> [L, T, E]
                B_cur = image.shape[0]
                N_tokens = indices.shape[0] // B_cur
                k = indices.shape[1]
                if token_routing_counts is None:
                    token_routing_counts = torch.zeros(num_moe_layers, N_tokens, num_experts, device=device)
                moe_idx = moe_block_to_layer[block_idx]
                indices_3d = indices.view(B_cur, N_tokens, k)
                one_hot = F.one_hot(indices_3d.long(), num_classes=num_experts)  # (B, N, k, E)
                token_routing_counts[moe_idx] += one_hot.sum(dim=(0, 2)).float()  # (N, E)

                # Ensure indices match mask length. If not, maybe dropped tokens? 
                # ViT usually keeps tokens unless specifically masked out in forward.
                if indices.shape[0] == fg_mask_flat.shape[0]:
                    fg_indices = indices[fg_mask_flat].flatten()
                    bg_indices = indices[~fg_mask_flat].flatten()
                    
                    fg_counts = torch.bincount(fg_indices, minlength=num_experts)
                    bg_counts = torch.bincount(bg_indices, minlength=num_experts)
                    
                    if dist.is_initialized():
                        dist.all_reduce(fg_counts, op=dist.ReduceOp.SUM)
                        dist.all_reduce(bg_counts, op=dist.ReduceOp.SUM)
                        
                    epoch_expert_counts_fg[block_idx] += fg_counts
                    epoch_expert_counts_bg[block_idx] += bg_counts

            if dist.is_initialized():
                # Sum expert counts from all processes
                dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)

            # Append the counts for the current batch to the collection for this block
            epoch_expert_counts_sum[block_idx] += expert_counts
            
            if block_idx == list(epoch_expert_counts_sum.keys())[-1]:
                total_tokens_epoch += expert_counts.sum().item()
            
            # Reset last_indices to free memory
            if hasattr(moe_layer, 'last_indices'):
                moe_layer.last_indices = None
            
            # reset count
            moe_layer.reset_counts()
            

    # After the epoch, calculate the final percentages
    final_expert_percentages_fg = {}
    final_expert_percentages_bg = {}

    if total_tokens_epoch > 0:
        for block_idx in epoch_expert_counts_fg.keys():
            total_fg = epoch_expert_counts_fg[block_idx].sum().item()
            total_bg = epoch_expert_counts_bg[block_idx].sum().item()
            
            if total_fg > 0:
                final_expert_percentages_fg[block_idx] = (epoch_expert_counts_fg[block_idx].float() / total_fg).cpu()
            else:
                final_expert_percentages_fg[block_idx] = torch.zeros(num_experts, dtype=torch.float)

            if total_bg > 0:
                final_expert_percentages_bg[block_idx] = (epoch_expert_counts_bg[block_idx].float() / total_bg).cpu()
            else:
                final_expert_percentages_bg[block_idx] = torch.zeros(num_experts, dtype=torch.float)
    else:
        for block_idx in range(num_blocks):
            final_expert_percentages_fg[block_idx] = torch.zeros(num_experts, dtype=torch.float)
            final_expert_percentages_bg[block_idx] = torch.zeros(num_experts, dtype=torch.float)

    # All-reduce and normalize per-token-position routing counts
    if token_routing_counts is not None:
        if dist.is_initialized():
            dist.all_reduce(token_routing_counts, op=dist.ReduceOp.SUM)
        # Normalize to per-position expert selection probability
        token_routing_probs = token_routing_counts / token_routing_counts.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    # After the epoch, save the collected expert counts to a pickle file
    if not dist.is_initialized() or dist.get_rank() == 0:
        _prefix = cfg.inference.output_prefix
        output_filename_fg = f'{_prefix}_foreground.pkl'
        output_filename_bg = f'{_prefix}_background.pkl'

        os.makedirs(os.path.dirname(os.path.abspath(output_filename_fg)), exist_ok=True)
        
        logger.info(f"Saving expert counts for the epoch to {output_filename_fg} and {output_filename_bg}...")
        with open(output_filename_fg, 'wb') as f:
            pickle.dump(final_expert_percentages_fg, f)
        with open(output_filename_bg, 'wb') as f:
            pickle.dump(final_expert_percentages_bg, f)
        logger.info("Done saving expert counts.")

        # Save per-token-position routing: [L, T, E]
        if token_routing_counts is not None:
            output_filename_routing = f'{_prefix}_token_routing.pkl'
            logger.info(f"Saving per-token-position routing to {output_filename_routing}...")
            with open(output_filename_routing, 'wb') as f:
                pickle.dump({
                    'counts': token_routing_counts.cpu(),  # [L, T, E] raw counts
                    'probs': token_routing_probs.cpu(),    # [L, T, E] normalized probabilities
                    'moe_block_indices': moe_block_indices,  # which block indices are MoE layers
                }, f)
            logger.info(f"Done saving token routing (shape: {token_routing_counts.shape}).")
 
    return


def inferencer(
    cfg: Any,
    model: Dict[str, torch.nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    logger: Optional[logging.Logger] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    wandb_run: Optional[Any] = None,
) -> float:
    """Inference for representation/experts distributions."""

    infer_one_epoch(
        cfg=cfg, 
        model=model, 
        loader=val_loader,
        logger=logger,
        device=device, 
        scaler=scaler,
        wandb_run=wandb_run, 
    )
    
    return