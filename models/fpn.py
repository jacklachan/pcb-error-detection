import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network that merges multi-scale features (1/4 to 1/32)
    into a uniform 256-channel representation. The top-down pathway with
    lateral connections aggregates context from coarse to fine scales,
    improving detection of tiny defects.
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        ])

    def forward(self, features):
        # features: [c2, c3, c4, c5]
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down: propagate coarse semantics to finer levels
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
