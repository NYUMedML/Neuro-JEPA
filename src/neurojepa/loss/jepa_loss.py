import torch
from neurojepa.masks.utils import apply_masks


def loss_fn(z, h, masks_pred, loss_exp=1.0, fg_map=None, bg_weight=1.0):
    """
    JEPA prediction loss with optional foreground-aware weighting.

    When *fg_map* (shape [B, num_patches], float with 1.0=foreground) is
    provided and *bg_weight* < 1.0, background tokens receive lower weight
    in the loss so the model focuses representation capacity on tissue.

    Weight per token:  w_i = bg_weight + (1 - bg_weight) * fg_map_i
      - foreground (fg=1): w = 1.0
      - background (fg=0): w = bg_weight
    Weights are normalized per-sample so the effective learning rate is
    independent of the foreground/background ratio.
    """
    # predictor will have returned only masked tokens for z
    h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]

    # Build per-token weights aligned with masks_pred
    use_weighting = fg_map is not None and bg_weight < 1.0
    if use_weighting:
        # fg_map - [B, num_patches], gather weights for each mask scale
        pred_fg = [apply_masks(fg_map.unsqueeze(-1), mi, concat=False)
                   for mi in masks_pred]  # list[ list[ [B, K, 1] ] ]

    loss, n = 0, 0
    for scale_idx, (zi, hi) in enumerate(zip(z, h)):
        for mask_idx, (zij, hij) in enumerate(zip(zi, hi)):
            per_token = torch.abs(zij - hij) ** loss_exp / loss_exp  # [B, K, D]
            per_token = per_token.mean(dim=-1)  # [B, K]

            if use_weighting:
                fg_vals = pred_fg[scale_idx][mask_idx].squeeze(-1)  # [B, K]
                w = bg_weight + (1.0 - bg_weight) * fg_vals        # [B, K]
                # Clamp denominator to bg_weight
                w = w / w.mean(dim=-1, keepdim=True).clamp(min=1e-6)
                loss += (per_token * w).mean()
            else:
                loss += per_token.mean()
            n += 1
    loss /= n
    return loss