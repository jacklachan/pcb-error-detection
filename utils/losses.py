import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Bipartite matching between predicted queries and ground-truth targets.
    Minimises a combined cost of classification, L1 box, and GIoU box.
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Q = outputs['pred_logits'].shape[:2]

        pred_logits = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [B*Q, C+1]
        pred_boxes = outputs['pred_boxes'].flatten(0, 1)                # [B*Q, 4]

        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_boxes = torch.cat([t['boxes'] for t in targets])

        if len(tgt_boxes) == 0:
            return [(torch.empty(0, dtype=torch.int64, device=pred_logits.device),
                     torch.empty(0, dtype=torch.int64, device=pred_logits.device))
                    for _ in targets]

        cost_class = -pred_logits[:, tgt_ids]
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes)
        )

        C = (self.cost_class * cost_class
             + self.cost_bbox * cost_bbox
             + self.cost_giou * cost_giou).view(B, Q, -1).cpu()

        sizes = [len(t['boxes']) for t in targets]
        indices, offset = [], 0
        for b, size in enumerate(sizes):
            if size > 0:
                cost_b = C[b, :, offset:offset + size].numpy()
                i, j = linear_sum_assignment(cost_b)
                indices.append((torch.as_tensor(i, dtype=torch.int64),
                                 torch.as_tensor(j, dtype=torch.int64)))
            else:
                indices.append((torch.empty(0, dtype=torch.int64),
                                 torch.empty(0, dtype=torch.int64)))
            offset += size

        return indices


class PCBDetectionLoss(nn.Module):
    """
    DETR-style set prediction loss:
      - Cross-entropy for classification (down-weighted no-object class)
      - L1 loss for bounding box regression
      - GIoU loss for box quality
    """

    def __init__(self, num_classes=6, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef  # penalise no-object less
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        # --- Classification loss ---
        pred_logits = outputs['pred_logits']
        B, Q = pred_logits.shape[:2]
        device = pred_logits.device

        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.int64, device=device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx):
                target_classes[i, src_idx] = targets[i]['labels'].to(device)[tgt_idx]

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)

        # --- Box losses (only on matched predictions) ---
        src_idx = self._batch_src_idx(indices, device)
        src_boxes = outputs['pred_boxes'][src_idx]
        tgt_boxes = torch.cat([
            t['boxes'].to(device)[j] for t, (_, j) in zip(targets, indices)
        ], dim=0)

        num_boxes = max(len(tgt_boxes), 1)

        if len(tgt_boxes) > 0:
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='sum') / num_boxes
            loss_giou = (1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)
            ))).sum() / num_boxes
        else:
            loss_bbox = src_boxes.sum() * 0
            loss_giou = src_boxes.sum() * 0

        total = loss_ce + 5.0 * loss_bbox + 2.0 * loss_giou
        return {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'total': total,
        }

    @staticmethod
    def _batch_src_idx(indices, device):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ]).to(device)
        src_idx = torch.cat([src for (src, _) in indices]).to(device)
        return batch_idx, src_idx
