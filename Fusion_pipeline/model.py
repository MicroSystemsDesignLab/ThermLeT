import torch
import torch.nn as nn
import torch.nn.functional as F
from submodule import PatchEmbed, Enc_Block, ResidualInResidualDenseBlock

class CoordConv2d(nn.Module):
    """
    CoordConv layer: concatenates normalized coordinate channels before applying Conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # in_channels + 2 for x and y coords
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        xs = torch.linspace(-1, 1, W, device=device)
        xs = xs.view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.linspace(-1, 1, H, device=device)
        ys = ys.view(1, 1, H, 1).expand(B, 1, H, W)
        x_aug = torch.cat([x, xs, ys], dim=1)
        return self.conv(x_aug)

class TemperatureTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224,224),
        patch_size=16,
        d_model=768,
        dropout=0.1,
        depth=12,
        nhead=12,
        dim_feedforward=2048,
        mlp_ratio=4.,
        qkv_bias=True,
        filters=64,
        out_chans=1,
        num_res_blocks=1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        H, W = img_size

        # Coordinate branch only
        self.coord_encoder = CoordConv2d(2, filters, kernel_size=3, padding=1)

        # Patch embeddings for power & coords
        self.patch_embed_power = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=1, embed_dim=d_model
        )
        self.patch_embed_coord = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=filters, embed_dim=d_model
        )

        num_patches = self.patch_embed_power.n_patches
        # Combined positional embedding for power & coord tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * 2, d_model))
        self.pos_drop  = nn.Dropout(p=dropout)

        # Transformer encoder blocks
        self.enc_blocks = nn.ModuleList([
            Enc_Block(
                dim=d_model,
                n_heads=nhead,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=dropout,
                attn_p=dropout,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Fold tokens back into spatial maps
        self.token_fold = nn.Fold(
            output_size=img_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Skip connections per modality
        self.conv_skip_power = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv_skip_coord = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        # Convs after folding tokens
        self.conv_fold_power = nn.Sequential(
            nn.Conv2d(d_model, filters, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_fold_coord = nn.Sequential(
            nn.Conv2d(d_model, filters, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Modality-specific fusion convs
        self.conv_fusion_power = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.conv_fusion_coord = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        # Residual-in-Residual Dense Blocks for refinement
        self.RRDB = nn.Sequential(*[
            ResidualInResidualDenseBlock(filters)
            for _ in range(num_res_blocks)
        ])

        # Final conv to output temperature map
        self.conv_out = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, out_chans, kernel_size=3, padding=1)
        )

    def forward(self, power, coords):
        """
        power:  [B,1,H,W]
        coords: [B,2,H,W]
        returns temperature map: [B,1,H,W]
        """
        B, _, H, W = power.shape

        # Encode coords -> [B,filters,H,W]
        coord_feats = self.coord_encoder(coords)

        # Patch embedding -> [B,P,d_model]
        x_p = self.patch_embed_power(power)
        x_c = self.patch_embed_coord(coord_feats)

        # Concatenate tokens + positional encoding -> [B,2P,d_model]
        x = torch.cat([x_p, x_c], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoding
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.norm(x)

        # Split back to tokens
        P = self.patch_embed_power.n_patches
        x_p, x_c = x[:, :P, :], x[:, P:, :]
        # [B,P,d_model] -> [B,d_model,P]
        x_p = x_p.transpose(1,2)
        x_c = x_c.transpose(1,2)

        # Proper fold usage: expand tokens into patches
        ph = pw = self.patch_size
        xp = x_p.unsqueeze(-1).repeat(1,1,1,ph*pw).reshape(B, -1, P)
        img_p = self.token_fold(xp)  # [B,d_model,H,W]
        xc = x_c.unsqueeze(-1).repeat(1,1,1,ph*pw).reshape(B, -1, P)
        img_c = self.token_fold(xc)

        # Post-fold conv + skip
        f_p = self.conv_fold_power(img_p) + self.conv_skip_power(power)
        f_c = self.conv_fold_coord(img_c) + self.conv_skip_coord(coord_feats)

        # Fusion & refinement
        fused   = self.conv_fusion_power(f_p) + self.conv_fusion_coord(f_c)
        refined = self.RRDB(fused)
        out     = self.conv_out(refined)
        return out


def build_temperature_model(args):
    return TemperatureTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        depth=args.num_layers,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        filters=args.filters,
        out_chans=1,
        num_res_blocks=args.num_res_blocks,
    )
