import torch
import torch.nn as nn

try:
    from torchvision.models import swin_t, Swin_T_Weights
    _SWIN_AVAILABLE = True
except ImportError:
    _SWIN_AVAILABLE = False


class SwinDualBackbone(nn.Module):
    """
    Shared Swin Transformer-Tiny backbone.
    Extracts multi-scale features from both template and test PCB images.

    Swin-T output channels per stage:
        stage 0  (1/4  resolution):  96 ch
        stage 1  (1/8  resolution): 192 ch
        stage 2  (1/16 resolution): 384 ch
        stage 3  (1/32 resolution): 768 ch
    """

    CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained=True):
        if not _SWIN_AVAILABLE:
            raise ImportError('torchvision >= 0.15 is required for Swin-T backbone.')
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        swin = swin_t(weights=weights)

        # torchvision Swin layout:
        #   features[0]: patch partition + linear embed  (stride 4, 96 ch)
        #   features[1]: stage 1 transformer blocks
        #   features[2]: patch merging  (stride 2, 192 ch)
        #   features[3]: stage 2 transformer blocks
        #   features[4]: patch merging  (stride 2, 384 ch)
        #   features[5]: stage 3 transformer blocks
        #   features[6]: patch merging  (stride 2, 768 ch)
        #   features[7]: stage 4 transformer blocks
        f = swin.features
        self.stage0 = nn.Sequential(f[0], f[1])   # 1/4,  96
        self.stage1 = nn.Sequential(f[2], f[3])   # 1/8, 192
        self.stage2 = nn.Sequential(f[4], f[5])   # 1/16, 384
        self.stage3 = nn.Sequential(f[6], f[7])   # 1/32, 768

    def _forward_single(self, x):
        # torchvision Swin outputs [B, H, W, C]; convert to [B, C, H, W]
        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return [s.permute(0, 3, 1, 2).contiguous() for s in [s0, s1, s2, s3]]

    def forward(self, template, test):
        return self._forward_single(template), self._forward_single(test)
