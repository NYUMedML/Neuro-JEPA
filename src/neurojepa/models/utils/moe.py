import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional, Literal

import torch.distributed as dist

global world_size, rank
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    return F.linear(x, weight, bias)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_routed_experts (int): Number of experts to route to.
        n_activated_experts (int): Number of experts activated.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
    """
    def __init__(self, 
                 dim: int, 
                 n_routed_experts: int,
                 n_activated_experts: int,
                 score_func = "sigmoid", 
                 route_scale = 1.0):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.score_func = score_func
        self.gate = nn.Linear(dim, n_routed_experts, bias=False)
        
        #self.bias = nn.Parameter(torch.empty(n_routed_experts))
        self.register_buffer("bias", torch.zeros(n_routed_experts))
        #self.bias = nn.Parameter(torch.zeros(n_routed_experts), requires_grad=False)
        self.route_scale = route_scale

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Routing weights, selected expert indices, and scores.
        """
        #scores = linear(x, self.weight)
        logits = self.gate(x)
        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = logits.sigmoid()
        
        # Save original scores for weight computation (gradients flow through here)
        original_scores = scores
        # Add bias ONLY for routing/selection (not for weight computation)
        scores = scores + self.bias
        
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            # Clamp denominator to prevent inf/nan when all top-k scores are ~0
            weights /= weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights *= self.route_scale
        
        # Store for analysis
        self.last_indices = indices.detach()
        return weights.type_as(x), indices, original_scores


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, 
                 dim: int, 
                 n_shared_experts: int,
                 n_routed_experts: int,
                 n_activated_experts: int,
                 moe_inter_dim: int,
                 score_func = "sigmoid", 
                 route_scale = 1.0,
                 act_layer = nn.GELU,
                 drop = 0.0,
                 ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = n_routed_experts
        self.gate = Gate(
            dim=dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            score_func=score_func,
            route_scale=route_scale
        )
        self.experts = nn.ModuleList([MLP(in_features=dim, hidden_features=moe_inter_dim, act_layer=act_layer, drop=drop) 
                                      if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(in_features=dim, hidden_features=n_shared_experts * moe_inter_dim, act_layer=act_layer, drop=drop)

        # self.counts = torch.zeros(self.n_routed_experts)
        self.register_buffer('counts', torch.zeros(self.n_routed_experts, dtype=torch.long))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.
            mask (Optional[torch.Tensor]): Boolean mask indicating valid tokens.
                                           True for valid tokens, False for padding/invalid.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices, original_scores = self.gate(x)
        y = torch.zeros_like(x)
        
        # Determine valid indices for counting
        flat_indices = indices.flatten()
        if mask is not None:
            # mask should be [B, N] or broadcastable to match x's batch dims
            # Flatten mask to align with x
            mask_flat = mask.view(-1)
            # Expand mask for top-k selections
            # indices is [B*N, k], mask is [B*N]
            mask_expanded = mask_flat.unsqueeze(-1).expand_as(indices).flatten()
            valid_indices = flat_indices[mask_expanded]
        else:
            valid_indices = flat_indices

        batch_counts = torch.bincount(valid_indices, minlength=self.n_routed_experts)
        self.counts += batch_counts.to(self.counts.device)

        # Store indices for external access
        self.last_indices = indices.detach()

        # Build a boolean mask for valid tokens to skip expert compute on invalid ones
        if mask is not None:
            valid_mask = mask.view(-1)  # [B*N], True = valid
        else:
            valid_mask = None

        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            if batch_counts[i] == 0:
                # Run a dummy forward so all expert params stay in the autograd
                # graph for DDP (required when using static_graph=True with MoE).
                y = y + 0.0 * expert(x[:1]).sum()
                continue
            idx, top = torch.where(indices == i)
            if valid_mask is not None:
                # Only run expert on valid tokens assigned to it
                valid_sel = valid_mask[idx]
                valid_idx = idx[valid_sel]
                valid_top = top[valid_sel]
                if valid_idx.numel() > 0:
                    y[valid_idx] += expert(x[valid_idx]) * weights[valid_idx, valid_top, None]
            else:
                y[idx] += expert(x[idx]) * weights[idx, top, None]

        if valid_mask is not None:
            # Only run shared experts on valid tokens
            valid_token_indices = valid_mask.nonzero(as_tuple=True)[0]
            if valid_token_indices.numel() > 0:
                shared_out = self.shared_experts(x[valid_token_indices])
                z = torch.zeros_like(x, dtype=shared_out.dtype)
                z[valid_token_indices] = shared_out
            else:
                z = torch.zeros_like(x)
            # Zero out scores for invalid tokens so downstream losses are clean
            original_scores = original_scores * valid_mask.unsqueeze(-1).float()
        else:
            z = self.shared_experts(x)

        return (y + z).view(shape), original_scores

    def reset_counts(self):
        """Reset expert usage counts. Call this after bias updates."""
        self.counts.zero_()


def calculate_vio(expert_counts):
    # Protect against empty or zero-count situations to avoid NaNs/inf during early steps.
    if expert_counts.numel() == 0:
        return [1.0, 1.0]
    total = expert_counts.sum()
    if total == 0:
        return [1.0, 1.0]

    avg_count = expert_counts.float().mean()
    if avg_count == 0:
        return [1.0, 1.0]

    min_violation = torch.min(expert_counts.float()) / avg_count 
    max_violation = torch.max(expert_counts.float()) / avg_count 
    
    return [min_violation.item(), max_violation.item()]


def calculate_vio_model(model, used_ddp=False, vit=False):
    """
    Calculate load balancing violations for MoE layers without updating bias.
    
    Args:
        model: The model containing MoE layers
        used_ddp: Whether using DistributedDataParallel
        vit: Whether model is a plain ViT (vs encoder with .backbone)
    """
    # Identify the core model (consistent with moe_bias_update)
    if used_ddp:
        model_noddp = model.module
    else:
        model_noddp = model
    
    backbone = model_noddp if vit else getattr(model_noddp, 'backbone', model_noddp)
    
    # Collect MoE layers
    moe_layers = [b.mlp for b in backbone.blocks if hasattr(b, 'mlp') and isinstance(b.mlp, MoE)]
    
    if not moe_layers:
        return 1.0, 1.0
    
    # Consolidate counts for single all_reduce
    all_counts = torch.stack([m.counts for m in moe_layers])
    
    if used_ddp and dist.is_initialized():
        dist.all_reduce(all_counts, op=dist.ReduceOp.SUM)
    
    total_minvio = 0.0
    total_maxvio = 0.0
    
    for i, moe_layer in enumerate(moe_layers):
        min_vio, max_vio = calculate_vio(all_counts[i])
        total_minvio += min_vio
        total_maxvio += max_vio
        moe_layer.reset_counts()

    n = len(moe_layers)
    return total_minvio / n, total_maxvio / n


def unpack_moe_params(moe_params):
    dim = moe_params.dim
    n_shared_experts = moe_params.n_shared_experts
    n_routed_experts = moe_params.n_routed_experts
    n_activated_experts = moe_params.n_activated_experts
    moe_inter_dim = moe_params.moe_inter_dim
    score_func = moe_params.score_func
    route_scale = moe_params.route_scale

    return dim, n_shared_experts, n_routed_experts, n_activated_experts, moe_inter_dim, score_func, route_scale


def moe_bias_update(model, update_rate, used_ddp=False, vit=False, bias_clip=0.3, use_proportional=False):
    """
    Optimized MoE auxiliary-loss-free bias update.
    Consolidates DDP communication and ensures zero-mean bias projection.
    
    Args:
        model: The model containing MoE layers
        update_rate: Base learning rate for bias updates
        used_ddp: Whether using DistributedDataParallel
        vit: Whether model is a plain ViT (vs encoder with .backbone)
        bias_clip: Maximum absolute value for bias (prevents runaway bias)
        use_proportional: If True, use proportional updates instead of sign-based
    """
    # Identify the core model
    if used_ddp:
        model_noddp = model.module
    else:
        model_noddp = model
        
    # Standard ViT vs. Encoder-style wrappers
    backbone = model_noddp if vit else getattr(model_noddp, 'backbone', model_noddp)
    
    # Collect all MoE layers to avoid redundant looping
    moe_layers = [b.mlp for b in backbone.blocks if hasattr(b, 'mlp') and isinstance(b.mlp, MoE)]
    
    if not moe_layers:
        return 0.0, 0.0

    # Consolidate expert counts for a single All-Reduce (Efficiency Boost)
    all_counts = torch.stack([m.counts for m in moe_layers]).long()
    
    if used_ddp and dist.is_initialized():
        dist.all_reduce(all_counts, op=dist.ReduceOp.SUM)
    
    # Convert to float after all-reduce for stability in calculations
    all_counts = all_counts.float() 
    
    total_minvio = 0.0
    total_maxvio = 0.0
    
    # Apply updates
    with torch.no_grad():
        for i, moe_layer in enumerate(moe_layers):
            layer_counts = all_counts[i]
            avg_count = layer_counts.mean()
            
            # Calculate violations for logging
            if avg_count > 0:
                rel_counts = layer_counts / avg_count
                total_minvio += rel_counts.min().item()
                total_maxvio += rel_counts.max().item()
                
                # Calculate Error: Positive means under-utilized
                error = avg_count - layer_counts
                
                if use_proportional:
                    # Proportional update with epsilon for stability
                    update = error / (avg_count + 1e-6)
                else:
                    # Sign-based update
                    update = torch.sign(error)
                
                # Zero-center the update itself (prevents drift)
                update = update - update.mean()
                
                # Apply update
                moe_layer.gate.bias.data += update_rate * update
                
                # Clipping
                if bias_clip > 0:
                    moe_layer.gate.bias.data.clamp_(-bias_clip, bias_clip)
                
                # Re-apply zero-mean projection after clipping
                # This ensures biases stay centered even after clipping
                moe_layer.gate.bias.data -= moe_layer.gate.bias.data.mean()
            else:
                # No tokens routed, reset violations to neutral
                total_minvio += 1.0
                total_maxvio += 1.0
            
            # Reset counts for the next iteration
            moe_layer.reset_counts()

    n = len(moe_layers)
    return total_minvio / n, total_maxvio / n
