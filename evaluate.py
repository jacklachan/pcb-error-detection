"""
Evaluation script — computes per-class AP, mAP@0.5, Precision, Recall, F1.

Usage:
    python evaluate.py --data_root path/to/DeepPCB --checkpoint output/best_model.pth
    python evaluate.py --data_root path/to/DeepPCB --checkpoint output/best_model.pth --iou_threshold 0.5
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DeepPCBDataset
from data.transforms import PCBTransform
from models.pcb_detector import PCBDefectDetector
from utils.box_ops import box_cxcywh_to_xyxy, box_iou
from utils.metrics import compute_map, DEFECT_CLASSES


def collect_outputs(model, loader, device, threshold=0.05):
    """Run inference and collect all predictions and ground truths."""
    all_preds, all_targets = [], []

    with torch.no_grad():
        for templates, tests, targets in tqdm(loader, desc='Collecting predictions'):
            templates = templates.to(device)
            tests     = tests.to(device)
            outputs   = model(templates, tests)

            for i, target in enumerate(targets):
                logits = outputs['pred_logits'][i].softmax(-1)
                boxes  = outputs['pred_boxes'][i]
                scores, labels = logits[:, :-1].max(-1)
                keep = scores > threshold  # low threshold — mAP sweeps all confidences

                all_preds.append({
                    'boxes':  boxes[keep].cpu(),
                    'labels': labels[keep].cpu(),
                    'scores': scores[keep].cpu(),
                })
                all_targets.append({
                    'boxes':  target['boxes'],
                    'labels': target['labels'],
                })

    return all_preds, all_targets


def compute_f1_at_threshold(all_preds, all_targets, threshold, iou_threshold):
    """Precision / Recall / F1 at a fixed confidence threshold."""
    tp = [0] * len(DEFECT_CLASSES)
    fp = [0] * len(DEFECT_CLASSES)
    fn = [0] * len(DEFECT_CLASSES)

    for preds, targets in zip(all_preds, all_targets):
        keep = preds['scores'] > threshold
        pred_boxes  = box_cxcywh_to_xyxy(preds['boxes'][keep])
        pred_labels = preds['labels'][keep]

        gt_boxes  = box_cxcywh_to_xyxy(targets['boxes'])
        gt_labels = targets['labels']
        matched   = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for pb, pl in zip(pred_boxes, pred_labels):
            hit = False
            if len(gt_boxes) > 0:
                ious, _ = box_iou(pb.unsqueeze(0), gt_boxes)
                best_iou, best_j = ious[0].max(0)
                if best_iou.item() >= iou_threshold and not matched[best_j]:
                    tp[pl.item()] += 1
                    matched[best_j] = True
                    hit = True
            if not hit:
                fp[pl.item()] += 1

        for j, gl in enumerate(gt_labels):
            if not matched[j]:
                fn[gl.item()] += 1

    return tp, fp, fn


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved = ckpt.get('args', {})

    model = PCBDefectDetector(
        backbone=saved.get('backbone', 'resnet50'),
        num_queries=saved.get('num_queries', 100),
        num_classes=6,
        pretrained=False,
        num_decoder_layers=saved.get('decoder_layers', 6),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    img_size = saved.get('img_size', 512)
    dataset  = DeepPCBDataset(
        args.data_root, split='test',
        transform=PCBTransform((img_size, img_size), train=False),
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                               torch.stack([x[1] for x in b]),
                               [x[2] for x in b]),
    )
    print(f'Evaluating on {len(dataset)} test pairs  (backbone={saved.get("backbone","resnet50")})\n')

    all_preds, all_targets = collect_outputs(model, loader, device, threshold=0.05)

    # --- mAP ---
    per_class_ap, map_score = compute_map(all_preds, all_targets, iou_threshold=args.iou_threshold)

    print(f'mAP@{args.iou_threshold:.2f}:')
    print(f'  {"Class":<20} {"AP":>8}')
    print('  ' + '-' * 30)
    for cls_name, ap in per_class_ap.items():
        marker = '  (no GT)' if ap != ap else ''  # NaN check
        val    = f'{ap:.4f}' if ap == ap else ' N/A '
        print(f'  {cls_name:<20} {val:>8}{marker}')
    print('  ' + '-' * 30)
    print(f'  {"mAP":<20} {map_score:>8.4f}\n')

    # --- P / R / F1 at fixed threshold ---
    tp, fp, fn = compute_f1_at_threshold(all_preds, all_targets,
                                          threshold=args.conf_threshold,
                                          iou_threshold=args.iou_threshold)
    hdr = f"  {'Class':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    print(f'Precision / Recall / F1  (conf > {args.conf_threshold}):')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for c, cls_name in enumerate(DEFECT_CLASSES):
        prec = tp[c] / (tp[c] + fp[c] + 1e-8)
        rec  = tp[c] / (tp[c] + fn[c] + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        print(f'  {cls_name:<20} {tp[c]:>5} {fp[c]:>5} {fn[c]:>5} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}')

    total_tp, total_fp, total_fn = sum(tp), sum(fp), sum(fn)
    prec = total_tp / (total_tp + total_fp + 1e-8)
    rec  = total_tp / (total_tp + total_fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    print('  ' + '-' * (len(hdr) - 2))
    print(f'  {"Overall":<20} {total_tp:>5} {total_fp:>5} {total_fn:>5} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',      required=True)
    parser.add_argument('--checkpoint',     required=True)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold',  type=float, default=0.5)
    args = parser.parse_args()
    main(args)
