"""
Mean Average Precision (mAP) computation for PCB defect detection.
Follows the VOC 2010+ standard: area under the precision-recall curve
computed via 11-point interpolation.
"""
import numpy as np
import torch
from collections import defaultdict

from .box_ops import box_iou

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']


def _voc_ap(recalls, precisions):
    """11-point interpolated AP."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        ap += (np.max(precisions[mask]) if mask.any() else 0.0) / 11.0
    return ap


def compute_map(all_predictions, all_targets, num_classes=6, iou_threshold=0.5):
    """
    Compute per-class AP and mAP.

    Args:
        all_predictions: list of dicts, each with:
            'boxes'  (N, 4) tensor  cx,cy,w,h normalised → converted to xyxy inside
            'labels' (N,)   int64 tensor
            'scores' (N,)   float tensor
        all_targets: list of dicts, each with:
            'boxes'  (M, 4) tensor  (same format)
            'labels' (M,)   int64 tensor
        num_classes: number of defect classes (6 for DeepPCB)
        iou_threshold: IoU threshold for a prediction to count as TP

    Returns:
        per_class_ap: dict {class_name: ap_value}
        mAP: float
    """
    from .box_ops import box_cxcywh_to_xyxy

    per_class_ap = {}

    for cls in range(num_classes):
        cls_preds = []          # (score, img_idx, box_xyxy)
        cls_gts = defaultdict(list)  # img_idx -> [box_xyxy, ...]

        for img_idx, (preds, targets) in enumerate(zip(all_predictions, all_targets)):
            pred_boxes_xyxy = box_cxcywh_to_xyxy(preds['boxes'])
            gt_boxes_xyxy   = box_cxcywh_to_xyxy(targets['boxes'])

            for box, lbl in zip(gt_boxes_xyxy, targets['labels']):
                if lbl.item() == cls:
                    cls_gts[img_idx].append(box)

            for box, lbl, score in zip(pred_boxes_xyxy, preds['labels'], preds['scores']):
                if lbl.item() == cls:
                    cls_preds.append((score.item(), img_idx, box))

        num_gt = sum(len(v) for v in cls_gts.values())
        if num_gt == 0:
            per_class_ap[DEFECT_CLASSES[cls]] = float('nan')
            continue

        if not cls_preds:
            per_class_ap[DEFECT_CLASSES[cls]] = 0.0
            continue

        cls_preds.sort(key=lambda x: -x[0])

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        matched = defaultdict(set)

        for k, (_, img_idx, pred_box) in enumerate(cls_preds):
            gts = cls_gts[img_idx]
            if not gts:
                fp[k] = 1
                continue

            gt_tensor = torch.stack(gts)
            ious, _ = box_iou(pred_box.unsqueeze(0), gt_tensor)
            best_iou, best_j = ious[0].max(0)

            if best_iou.item() >= iou_threshold and best_j.item() not in matched[img_idx]:
                tp[k] = 1
                matched[img_idx].add(best_j.item())
            else:
                fp[k] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls    = cum_tp / (num_gt + 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

        per_class_ap[DEFECT_CLASSES[cls]] = _voc_ap(recalls, precisions)

    valid_aps = [v for v in per_class_ap.values() if not np.isnan(v)]
    map_score = float(np.mean(valid_aps)) if valid_aps else 0.0
    return per_class_ap, map_score
