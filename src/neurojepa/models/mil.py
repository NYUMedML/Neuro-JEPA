import torch
import torch.nn as nn
from functools import partial
from typing import List, Optional


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Optional[List[int]] = None):
        super().__init__()
        dims = [in_dim] + (hidden_dims or []) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def segment_softmax(scores: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax over variable-length segments defined by cu_seqlens.

    Args:
        scores: [N] raw scores
        cu_seqlens: [B+1] cumulative sequence lengths

    Returns:
        [N] softmax-normalized attention weights
    """
    orig_dtype = scores.dtype
    scores = scores.float()  # upcast for numerical stability
    N = scores.size(0)
    B = len(cu_seqlens) - 1
    device = scores.device

    # Build index mapping each token -> its bag index
    instance_ids = torch.zeros(N, dtype=torch.long, device=device)
    for i in range(B):
        instance_ids[cu_seqlens[i]:cu_seqlens[i + 1]] = i

    # Numerically stable softmax per segment
    # Subtract per-segment max
    seg_max = torch.full((B,), float('-inf'), device=device)
    seg_max.scatter_reduce_(0, instance_ids, scores, reduce='amax', include_self=True)
    scores_stable = scores - seg_max[instance_ids]

    exp_scores = torch.exp(scores_stable)

    seg_sum = torch.zeros(B, device=device)
    seg_sum.scatter_add_(0, instance_ids, exp_scores)

    attention = exp_scores / seg_sum[instance_ids]
    return attention.to(orig_dtype), instance_ids


def segment_sum(values: torch.Tensor, instance_ids: torch.Tensor, B: int) -> torch.Tensor:
    """Sum values per segment."""
    out = torch.zeros(B, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, instance_ids, values)
    return out


def trunc_normal_(tensor, std=0.02):
    """Truncated normal initialization."""
    nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)


class ClassifyThenAggregate(nn.Module):
    """
    Classify-Then-Aggregate Multiple Instance Learning.

    Combines attention-weighted pooling with patch-level predictions. The final
    bag-level prediction is the attention-weighted sum of patch predictions.

    Args:
        dim (int): Input feature dimension
        hidden_dim (int, optional): Hidden dimension for attention computation.
                                    Defaults to dim.
        W_out (int): Number of output classes. Defaults to 1.
        mlp_hidden_dims (List[int], optional): Hidden dimensions for prediction MLP.
        mlp_out_dim (int, optional): Output dimension for MLP. Defaults to W_out.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        init_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        use_gating (bool): Whether to use gating mechanism. Defaults to True.
        use_norm (bool): Whether to apply normalization. Defaults to False.
        use_output_bias_scale (bool): Whether to apply learnable bias/scale to output.
                                      Defaults to True.
        norm_layer: Normalization layer constructor. Defaults to LayerNorm.

    Example:
        >>> import torch
        >>> from neurovfm.models import ClassifyThenAggregate
        >>>
        >>> mil = ClassifyThenAggregate(dim=768, W_out=2, mlp_hidden_dims=[512, 256])
        >>>
        >>> features = torch.randn(50, 768)
        >>> cu_seqlens = torch.tensor([0, 20, 35, 50])
        >>>
        >>> logits = mil(features, cu_seqlens=cu_seqlens)  # [3, 2]
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        W_out: int = 1,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_out_dim: Optional[int] = None,
        drop_rate: float = 0.0,
        init_std: float = 0.02,
        use_gating: bool = True,
        use_norm: bool = False,
        use_output_bias_scale: bool = True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_features = self.dim = dim
        if hidden_dim is None:
            hidden_dim = dim

        # Standard nn.Linear replaces FusedDense (same interface, no flash_attn needed)
        self.attention_V = nn.Linear(dim, hidden_dim, bias=True)
        self.gating_V = nn.Linear(dim, hidden_dim, bias=True) if use_gating else None
        self.W = nn.Linear(hidden_dim, W_out, bias=True)
        self.W_out = W_out

        self.dropout = nn.Dropout(p=drop_rate)

        self.norm_attn = norm_layer(dim) if use_norm else None
        self.norm_mlp = norm_layer(dim) if use_norm else None

        if mlp_out_dim is None:
            mlp_out_dim = W_out
        self.mlp = MLP(in_dim=dim, out_dim=mlp_out_dim, hidden_dims=mlp_hidden_dims)

        self.use_output_bias_scale = use_output_bias_scale
        if use_output_bias_scale:
            self.output_bias = nn.Parameter(torch.zeros(W_out))
            self.output_scale = nn.Parameter(torch.ones(W_out))

        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        media: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            media (torch.Tensor): Input features [N, dim]
            cu_seqlens (torch.Tensor): Cumulative sequence lengths [B+1]
            max_seqlen (int, optional): Unused, kept for API compatibility.
            return_logits (bool): If True, also return attention weights and patch logits.

        Returns:
            torch.Tensor: Bag-level predictions [B, W_out]
            OR Tuple: (output, attention_weights, patch_logits) if return_logits=True
        """
        # Validate cu_seqlens
        expected_tokens = cu_seqlens[-1].item()
        actual_tokens = media.size(0)
        if expected_tokens != actual_tokens:
            raise RuntimeError(
                f"ClassifyThenAggregate: cu_seqlens[-1]={expected_tokens} != "
                f"media.size(0)={actual_tokens}. Batch has inconsistent sizes."
            )

        B = len(cu_seqlens) - 1

        # Optional pre-norm
        media_attn = self.norm_attn(media) if self.norm_attn is not None else media
        media_mlp = self.norm_mlp(media) if self.norm_mlp is not None else media

        media_attn = self.dropout(media_attn)
        media_mlp = self.dropout(media_mlp)

        # Attention scores: [N, W_out]
        attention_features = torch.tanh(self.attention_V(media_attn))
        if self.gating_V is not None:
            gating_features = torch.sigmoid(self.gating_V(media_attn))
            score_features = attention_features * gating_features
        else:
            score_features = attention_features
        scores = self.W(score_features)  # [N, W_out]

        # Patch-level predictions: [N, mlp_out_dim]
        patch_logits = self.mlp(media_mlp)

        # Aggregate per class
        attention_weights = torch.zeros_like(scores)  # [N, W_out]
        class_outputs = []

        for c in range(self.W_out):
            attn_c, instance_ids = segment_softmax(scores[:, c], cu_seqlens)
            attention_weights[:, c] = attn_c

            pl_c = patch_logits[:, c] if patch_logits.size(1) > 1 else patch_logits.squeeze(-1)
            class_out = segment_sum(attn_c * pl_c, instance_ids, B)  # [B]
            class_outputs.append(class_out)

        output = torch.stack(class_outputs, dim=1)  # [B, W_out]

        if self.use_output_bias_scale:
            output = output * self.output_scale + self.output_bias

        if return_logits:
            return output, attention_weights, patch_logits
        return output