"""
Print model architecture summary and parameter counts.

Usage:
    python model_info.py
    python model_info.py --backbone swin_t
"""
import argparse
import torch
from models.pcb_detector import PCBDefectDetector


def count_params(module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main(args):
    model = PCBDefectDetector(
        backbone=args.backbone,
        num_queries=100,
        num_classes=6,
        pretrained=False,
    )

    sections = {
        'Backbone':         model.backbone,
        'Fusion (C2 diff)': model.fusion_c2,
        'Fusion (C3 diff)': model.fusion_c3,
        'Fusion (C4 attn)': model.fusion_c4,
        'Fusion (C5 attn)': model.fusion_c5,
        'FPN':              model.fpn,
        'Detection Head':   model.head,
    }

    print(f'\nPCB Defect Detector  —  backbone={args.backbone}')
    print('=' * 52)
    print(f'  {"Module":<22} {"Total params":>14} {"Trainable":>12}')
    print('-' * 52)
    for name, mod in sections.items():
        tot, train = count_params(mod)
        print(f'  {name:<22} {tot/1e6:>11.2f}M {train/1e6:>9.2f}M')
    print('-' * 52)
    tot, train = count_params(model)
    print(f'  {"TOTAL":<22} {tot/1e6:>11.2f}M {train/1e6:>9.2f}M')
    print('=' * 52)

    # Memory estimate for a single forward pass at 512x512
    template = torch.randn(1, 3, 512, 512)
    test     = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(template, test)
    print(f'\nForward pass (1 × 512×512 pair):')
    print(f'  pred_logits : {list(out["pred_logits"].shape)}')
    print(f'  pred_boxes  : {list(out["pred_boxes"].shape)}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50', choices=['resnet50', 'swin_t'])
    args = parser.parse_args()
    main(args)
