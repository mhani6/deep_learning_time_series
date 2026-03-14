"""
InceptionTime baseline — no pretraining, trains from scratch on LSST.
Reference: Ismail Fawaz et al., 2020.
"""

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, nb_filters=32, kernel_sizes=(9, 19, 39), bottleneck=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False) \
            if in_channels > 1 else None

        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck if in_channels > 1 else in_channels,
                      nb_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )
        self.bn  = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        inp = x
        if self.bottleneck:
            x = self.bottleneck(x)
        outs = [conv(x) for conv in self.convs] + [self.maxpool_conv(inp)]
        out  = torch.cat(outs, dim=1)
        return self.act(self.bn(out))


class InceptionTime(nn.Module):
    def __init__(self, n_channels, n_classes, nb_filters=32, depth=6):
        super().__init__()
        out_channels = nb_filters * 4  # 4 branches per block

        blocks = []
        residuals = []
        in_ch = n_channels
        res_in_ch = n_channels

        for i in range(depth):
            blocks.append(InceptionBlock(in_ch, nb_filters))
            in_ch = out_channels
            if i % 3 == 2:
                residuals.append(nn.Sequential(
                    nn.Conv1d(res_in_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                ))
                res_in_ch = out_channels

        self.blocks    = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList([r for r in residuals if r is not None])
        self.depth     = depth
        self.act       = nn.ReLU()
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(out_channels, n_classes)

        self._res_idx = [i for i in range(depth) if i % 3 == 2]

    def forward(self, x):
        # x: (B, T, C) → (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        res = x
        res_count = 0

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self._res_idx:
                shortcut = self.residuals[res_count](res)
                x = self.act(x + shortcut)
                res = x
                res_count += 1

        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


def build_baseline(n_channels, n_classes, device):
    model = InceptionTime(n_channels=n_channels, n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"InceptionTime baseline — {n_params:,} trainable parameters")
    return model
