"""
Dataset explorer — visualise sample PCB pairs with ground-truth annotations
and print dataset statistics.

Usage:
    python explore_dataset.py --data_root path/to/DeepPCB
    python explore_dataset.py --data_root path/to/DeepPCB --n 8 --split train
"""
import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import DeepPCBDataset, DEFECT_CLASSES
from data.transforms import PCBTransform

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])
COLORS = ['#FF4444', '#44DD44', '#4488FF', '#FFDD00', '#FF44FF', '#44FFFF']


def denorm(t):
    return np.clip(t.permute(1, 2, 0).numpy() * _STD + _MEAN, 0, 1)


def draw_boxes(ax, img, boxes, labels, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    H, W = img.shape[:2]
    for box, label in zip(boxes, labels):
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        w  = bw * W
        h  = bh * H
        color = COLORS[label % len(COLORS)]
        ax.add_patch(patches.Rectangle((x1, y1), w, h,
                                        linewidth=2, edgecolor=color, facecolor='none'))
        ax.text(x1, y1 - 4, DEFECT_CLASSES[label], color=color,
                fontsize=7, fontweight='bold')


def show_samples(dataset, n, save_path):
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        template, test, target = dataset[idx]
        boxes  = target['boxes'].numpy()
        labels = target['labels'].numpy()
        draw_boxes(axes[row][0], denorm(template), [], [], 'Template (defect-free)')
        draw_boxes(axes[row][1], denorm(test),     boxes, labels,
                   f'Test — {len(boxes)} defect(s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f'Sample grid saved to {save_path}')
    plt.close()


def print_stats(dataset):
    label_counts = Counter()
    boxes_per_image = []
    empty = 0

    for _, _, target in dataset:
        labels = target['labels'].tolist()
        label_counts.update(labels)
        boxes_per_image.append(len(labels))
        if len(labels) == 0:
            empty += 1

    total = len(dataset)
    print(f'\n{"="*50}')
    print(f'Dataset statistics  ({total} image pairs)')
    print(f'{"="*50}')
    print(f'  Images with no defects : {empty}  ({100*empty/total:.1f}%)')
    print(f'  Avg defects per image  : {np.mean(boxes_per_image):.2f}')
    print(f'  Max defects in one img : {max(boxes_per_image)}')
    print(f'\n  Per-class counts:')
    for cls_idx, cls_name in enumerate(DEFECT_CLASSES):
        count = label_counts[cls_idx]
        bar   = '█' * int(30 * count / max(label_counts.values(), default=1))
        print(f'    {cls_name:<18} {count:>5}  {bar}')
    print(f'{"="*50}\n')


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = DeepPCBDataset(
        args.data_root, split=args.split,
        transform=PCBTransform((512, 512), train=False),
    )

    print_stats(dataset)
    show_samples(
        dataset, n=args.n,
        save_path=os.path.join(args.output_dir, f'samples_{args.split}.png'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  required=True)
    parser.add_argument('--split',      default='train', choices=['train', 'test'])
    parser.add_argument('--n',          type=int, default=4, help='Number of sample pairs to show')
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()
    main(args)
