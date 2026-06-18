import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class DualBackbone(nn.Module):
    """
    Shared ResNet-50 backbone that extracts multi-scale features from
    both the template (defect-free) and test (manufactured) PCB images.
    Weight sharing ensures the same feature space for cross-attention.
    """

    CHANNELS = [256, 512, 1024, 2048]  # C2, C3, C4, C5

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # C2: 1/4,  256 ch
        self.layer2 = backbone.layer2  # C3: 1/8,  512 ch
        self.layer3 = backbone.layer3  # C4: 1/16, 1024 ch
        self.layer4 = backbone.layer4  # C5: 1/32, 2048 ch

    def _forward_single(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

    def forward(self, template, test):
        return self._forward_single(template), self._forward_single(test)
