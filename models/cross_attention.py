import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention where test features (queries) attend to template features
    (keys/values) to learn structural inconsistencies.

    Applied at coarse scales (C4: 1/16, C5: 1/32) where spatial resolution
    is small enough for full attention to be memory-efficient.
    """

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, template_feat, test_feat):
        B, C, H, W = template_feat.shape
        t = template_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        s = test_feat.flatten(2).permute(0, 2, 1)      # [B, HW, C]

        # Test queries attend to template keys/values
        attn_out, _ = self.attn(s, t, t)
        s = self.norm1(s + attn_out)
        s = self.norm2(s + self.ffn(s))

        return s.permute(0, 2, 1).reshape(B, C, H, W)


class DifferenceFusion(nn.Module):
    """
    Lightweight fusion for fine scales (C2: 1/4, C3: 1/8).
    Full attention on these scales would require O((H*W)^2) memory.
    Uses element-wise difference to capture local structural deviations.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, template_feat, test_feat):
        diff = test_feat - template_feat
        return self.conv(torch.cat([test_feat, diff], dim=1))
