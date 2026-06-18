import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .box_ops import box_cxcywh_to_xyxy

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']
COLORS = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', '#FF44FF', '#44FFFF']

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denorm(tensor):
    img = tensor.cpu().permute(1, 2, 0).numpy()
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)


def visualize_predictions(template, test, outputs, targets=None,
                           threshold=0.5, save_path=None):
    """
    Draw predicted and optionally ground-truth boxes on the test PCB image.
    """
    logits = outputs['pred_logits'][0].softmax(-1)
    boxes = outputs['pred_boxes'][0]
    scores, labels = logits[:, :-1].max(-1)
    keep = scores > threshold

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(_denorm(template[0]))
    axes[0].set_title('Template (Defect-Free)', fontsize=13)
    axes[0].axis('off')

    ax = axes[1]
    ax.imshow(_denorm(test[0]))
    ax.set_title('Test PCB — Predictions', fontsize=13)
    ax.axis('off')

    H, W = test.shape[-2:]
    pred_boxes_xyxy = box_cxcywh_to_xyxy(boxes[keep]).cpu().numpy()
    pred_boxes_xyxy[:, [0, 2]] *= W
    pred_boxes_xyxy[:, [1, 3]] *= H

    for box, label, score in zip(pred_boxes_xyxy,
                                  labels[keep].cpu().numpy(),
                                  scores[keep].cpu().numpy()):
        color = COLORS[label % len(COLORS)]
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f'{DEFECT_CLASSES[label]} {score:.2f}',
                color=color, fontsize=8, fontweight='bold')

    if targets is not None:
        gt_boxes = box_cxcywh_to_xyxy(targets[0]['boxes']).numpy()
        gt_boxes[:, [0, 2]] *= W
        gt_boxes[:, [1, 3]] *= H
        for box, label in zip(gt_boxes, targets[0]['labels'].numpy()):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor='white',
                                      facecolor='none', linestyle='--')
            ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved to {save_path}')
    else:
        plt.show()
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'Saved training curves to {save_path}')
