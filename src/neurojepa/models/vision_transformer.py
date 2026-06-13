# Adjusted from VJEPA2 https://github.com/facebookresearch/vjepa2/blob/main/src/models/vision_transformer.py

import math
from functools import partial

import torch
import torch.nn as nn

from neurojepa.masks.utils import apply_masks
from neurojepa.models.utils.modules import Block
from neurojepa.models.utils.patch_embed import PatchEmbed3D
from neurojepa.models.utils.pos_embs import get_3d_sincos_pos_embed
from neurojepa.utils.tensors import trunc_normal_


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224, 64),
        patch_size=(16, 16, 4),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        use_rope=False,
        use_moe=False,
        moe_params=None,
        handle_nonsquare_inputs=True,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        
        self.use_moe = use_moe
        
        if type(img_size) is int:
            img_size = (img_size, img_size, img_size)
        self.img_height, self.img_width, self.img_depth = img_size
        
        if type(patch_size) is int:
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        
        self.use_activation_checkpointing = use_activation_checkpointing
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Tokenize pixels with convolution
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        # Position embedding
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        if self.use_rope:
            self.pos_embed = None
            
        # Attention Blocks
        self.moe_layer_indices = moe_params.moe_layer_indices if \
            (use_moe and moe_params is not None) else []
        print(f"MOE layer indices: {self.moe_layer_indices}")
        # create a list for all layers if use_moe is True
        #self.moe_layers_idx = range(depth) if use_moe else []
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                    use_moe=use_moe if i in self.moe_layer_indices else False,
                    moe_params=moe_params,
                    grid_size=img_size[0] // patch_size[0],
                    grid_depth=img_size[2] // patch_size[2],
                    use_rope=use_rope,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

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
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if not self.use_moe or layer_id not in self.moe_layer_indices:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
            else:
                for expert in layer.mlp.experts:
                    if expert is not None:
                        rescale(expert.fc2.weight.data, layer_id + 1)
                rescale(layer.mlp.shared_experts.fc2.weight.data, layer_id + 1)
                
    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}
    
    def forward(self, x, masks=None, attn_mask=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove) — used during JEPA pre-training.
        :param attn_mask: optional bool tensor [B, N] (True = foreground/valid token).
                          Reshaped to [B, 1, 1, N] and passed to every self-attention block as
                          a key-mask so background tokens are suppressed in softmax.
                          Not supported together with token-removal ``masks`` (JEPA mode).
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Tokenize input
        if x.ndim == 5:
            _, _, H, W, D = x.shape
            D_patches = D // self.patch_size[2]
        else:
            raise ValueError("Input must be a 5D tensor")
            
        H_patches = H // self.patch_size[0]
        W_patches = W // self.patch_size[1]
        if not self.handle_nonsquare_inputs:
            D_patches = H_patches = W_patches = None

        x = self.patch_embed(x)

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        # Fwd prop
        outs = []
        outs_moe_scores = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, x_moe_scores = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, None, D_patches=D_patches, H_patches=H_patches, W_patches=W_patches, use_reentrant=False
                )
            else:
                x, x_moe_scores = blk(x, mask=masks, attn_mask=None, D_patches=D_patches, H_patches=H_patches, W_patches=W_patches)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))
            if self.use_moe:
                outs_moe_scores.append(x_moe_scores)

        if self.out_layers is not None:
            return outs, outs_moe_scores

        if self.norm is not None:
            x = self.norm(x)

        return x, outs_moe_scores

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        # If pos_embed already correct size, just return
        _, _, H, W, D = x.shape
        if H == self.img_height and W == self.img_width and D == self.img_depth:
            return pos_embed

        # Just chop off last N tokens of positional embedding
        elif H == self.img_height and W == self.img_width and D < self.img_depth:
            new_N = int((H // self.patch_size[0]) * (W // self.patch_size[1]) * (D // self.patch_size[2]))
            return pos_embed[:, :new_N, :]

        # Convert depth, height, width of input to be measured in patches
        # instead of pixels/frames
        H = H // self.patch_size[0]
        W = W // self.patch_size[1]
        D = D // self.patch_size[2]

        # Compute the initialized shape of the positional embedding measured
        # in patches
        N_h = self.img_height // self.patch_size[0]
        N_w = self.img_width // self.patch_size[1]
        N_d = self.img_depth // self.patch_size[2]
        assert N_h * N_w * N_d == N, "Positional embedding initialized incorrectly"

        # Compute scale factor for 3D-spatio interpolation
        scale_factor = (H / N_h, W / N_w, D / N_d)

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, N_h, N_w, N_d, dim).permute(0, 4, 1, 2, 3),
            scale_factor=scale_factor,
            mode="trilinear",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed
        
        
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

VIT_EMBED_DIMS = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
}