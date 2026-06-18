import torch
import torch.nn as nn

from .backbone import DualBackbone
from .swin_backbone import SwinDualBackbone
from .cross_attention import CrossAttentionFusion, DifferenceFusion
from .fpn import FPN
from .detection_head import TransformerDetectionHead

_BACKBONES = {
    'resnet50': DualBackbone,
    'swin_t':   SwinDualBackbone,
}


class PCBDefectDetector(nn.Module):
    """
    Full pipeline: Dual Backbone → Cross-Attention Fusion → FPN → Transformer Detection Head.

    Architecture from report:
    1. Shared backbone (ResNet-50 or Swin-T) extracts multi-scale features.
    2. Cross-attention at coarse scales (C4, C5) learns structural inconsistencies
       between template and test PCBs. Difference fusion at fine scales (C2, C3)
       captures local deviations without the memory cost of full attention.
    3. FPN merges all scales for multi-scale defect awareness.
    4. DETR-style decoder predicts defect bounding boxes and class labels.

    Args:
        backbone: 'resnet50' (default) or 'swin_t'
        num_queries: number of detection queries (100 recommended)
        num_classes: 6 for DeepPCB (open, short, mousebite, spur, pinhole, spurious_copper)
    """

    def __init__(self, backbone='resnet50', num_queries=100, num_classes=6,
                 pretrained=True, fpn_out_channels=256, num_decoder_layers=6):
        super().__init__()

        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {list(_BACKBONES)}, got '{backbone}'")

        self.backbone = _BACKBONES[backbone](pretrained=pretrained)
        channels = self.backbone.CHANNELS  # [C2, C3, C4, C5]

        # Fine scales: lightweight difference fusion (full attention too expensive here)
        self.fusion_c2 = DifferenceFusion(channels[0])
        self.fusion_c3 = DifferenceFusion(channels[1])

        # Coarse scales: full cross-attention
        self.fusion_c4 = CrossAttentionFusion(channels[2], num_heads=8)
        self.fusion_c5 = CrossAttentionFusion(channels[3], num_heads=8)

        self.fpn = FPN(in_channels=channels, out_channels=fpn_out_channels)

        self.head = TransformerDetectionHead(
            d_model=fpn_out_channels,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
        )

    def forward(self, template, test):
        template_feats, test_feats = self.backbone(template, test)

        fused = [
            self.fusion_c2(template_feats[0], test_feats[0]),
            self.fusion_c3(template_feats[1], test_feats[1]),
            self.fusion_c4(template_feats[2], test_feats[2]),
            self.fusion_c5(template_feats[3], test_feats[3]),
        ]

        return self.head(self.fpn(fused))
