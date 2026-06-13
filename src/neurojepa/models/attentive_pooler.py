# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch
import torch.nn as nn

from neurojepa.models.utils.modules import Block, CrossAttention, CrossAttentionBlock
from neurojepa.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, attn_mask=None):
        """
        :param x: input features [B, N, D]
        :param attn_mask: optional boolean mask [B, N]. True = valid, False = masked out.
                          Applied as a key-mask in the self-attention pre-processing blocks
                          (prevents attending to background keys) and as a cross-attention
                          key-mask (prevents query tokens from attending to background tokens).
        """
        # if moe backbone, x is a tuple (features, router logits)
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]

        # Convert [B, N] bool mask -> [B, 1, 1, N] for SDPA key-masking in self-attention.
        # True = valid key, False = background key to suppress.
        sa_key_mask = None
        if attn_mask is not None and self.blocks is not None:
            sa_key_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    # positional args: (x, mask, attn_mask, ...)
                    x, _ = torch.utils.checkpoint.checkpoint(blk, x, None, sa_key_mask, use_reentrant=False)
                else:
                    x, _ = blk(x, attn_mask=sa_key_mask)
                    
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x, attn_mask=attn_mask)
        return q


class AttentiveClassifier(nn.Module):
    """Attentive Classifier"""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x, attn_mask=None):
        x = self.pooler(x, attn_mask=attn_mask).squeeze(1)
        x = self.linear(x)
        return x
    
    
class MultiAttentive(nn.Module):
    def __init__(self, embed_dim, num_classes, device):
        super().__init__()
        self.classifiers = nn.ModuleList([
            AttentiveClassifier(
                embed_dim=embed_dim,
                num_heads=16,
                depth=4,
                num_classes=1,
            ).to(device)
            for _ in range(num_classes)
        ])

    def forward(self, x, attn_mask=None):
        """
        :param x: backbone features [B, N, D] (or tuple from MoE backbone)
        :param attn_mask: optional bool tensor [B, N], True = foreground/valid patch.
                          Passed to each AttentiveClassifier so cross-attention query
                          tokens only attend to foreground patches.
        """
        return [m(x, attn_mask=attn_mask) for m in self.classifiers]