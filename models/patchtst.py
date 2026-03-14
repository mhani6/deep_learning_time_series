"""
PatchTST adapted for multivariate time series classification.

Original paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
Nie et al., ICLR 2023 — https://arxiv.org/abs/2211.14730

Key idea: split each channel into non-overlapping patches, treat patches as tokens,
apply a Transformer encoder, then classify from the [CLS] token (or global avg pool).
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Splits a time series of shape (B, T, C) into patches and projects each patch.

    For each channel independently:
      - Divide T into num_patches = (T - patch_len) // stride + 1 patches
      - Linearly project each patch to d_model
    """
    def __init__(self, seq_len, patch_len, stride, d_model, n_channels, dropout=0.1):
        super().__init__()
        self.patch_len  = patch_len
        self.stride     = stride
        self.n_channels = n_channels

        num_patches = (seq_len - patch_len) // stride + 1
        self.num_patches = num_patches

        # One linear projection shared across all channels
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout    = nn.Dropout(dropout)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        # Unfold T into patches for each channel
        # → (B, C, num_patches, patch_len)
        x = x.permute(0, 2, 1)                        # (B, C, T)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B, C, num_patches, patch_len)

        # Project patches: treat (B*C) as batch
        B, C, N, P = x.shape
        x = x.reshape(B * C, N, P)                    # (B*C, num_patches, patch_len)
        x = self.projection(x)                         # (B*C, num_patches, d_model)
        x = x + self.pos_embedding                     # add positional encoding
        x = self.dropout(x)
        return x, B, C, N                              # return B,C,N for reshaping later


class PatchTSTClassifier(nn.Module):
    """
    Full PatchTST model for classification.

    Architecture:
      1. PatchEmbedding  → patches per channel
      2. Transformer encoder (channel-independent)
      3. Global average pooling over patches
      4. Concatenate all channel representations
      5. MLP classification head
    """
    def __init__(
        self,
        seq_len,
        n_channels,
        n_classes,
        patch_len=4,       # patch size (seq_len=36 → patch_len=4 gives 9 patches)
        stride=4,          # non-overlapping patches
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        head_dropout=0.2,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            seq_len, patch_len, stride, d_model, n_channels, dropout
        )
        num_patches = self.patch_embed.num_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head: flatten all channel representations
        self.norm       = nn.LayerNorm(d_model)
        self.head_drop  = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(d_model * n_channels, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, C)
        patches, B, C, N = self.patch_embed(x)   # (B*C, N, d_model)

        # Transformer encoder
        out = self.transformer(patches)            # (B*C, N, d_model)
        out = self.norm(out)

        # Global average pooling over patches
        out = out.mean(dim=1)                      # (B*C, d_model)

        # Reshape back to (B, C, d_model) then flatten channels
        out = out.reshape(B, C, -1)                # (B, C, d_model)
        out = out.reshape(B, -1)                   # (B, C * d_model)

        out = self.head_drop(out)
        logits = self.classifier(out)              # (B, n_classes)
        return logits


def build_model(seq_len, n_channels, n_classes, device):
    """Build PatchTST with sensible defaults for LSST (T=36, C=6)."""
    model = PatchTSTClassifier(
        seq_len=seq_len,
        n_channels=n_channels,
        n_classes=n_classes,
        patch_len=4,
        stride=4,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        head_dropout=0.2,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchTST — {n_params:,} trainable parameters")
    return model
