"""
Run inference on a single template/test PCB image pair.

Usage:
    python inference.py --template path/to/temp.jpg --test path/to/test.jpg \
                        --checkpoint output/best_model.pth --output result.png
"""
import argparse
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from models.pcb_detector import PCBDefectDetector
from utils.box_ops import box_cxcywh_to_xyxy

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']
COLORS = [(255, 68, 68), (68, 255, 68), (68, 68, 255),
          (255, 255, 68), (255, 68, 255), (68, 255, 255)]

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])


def preprocess(path, size):
    img = Image.open(path).convert('RGB')
    t = _NORMALIZE(TF.to_tensor(TF.resize(img, size)))
    return t.unsqueeze(0), img


def draw_boxes(image, boxes_xyxy, labels, scores, size):
    draw_img = image.resize(size)
    draw = ImageDraw.Draw(draw_img)
    W, H = size

    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1 = box[0] * W
        y1 = box[1] * H
        x2 = box[2] * W
        y2 = box[3] * H
        color = COLORS[label % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f'{DEFECT_CLASSES[label]}: {score:.2f}'
        draw.rectangle([x1, y1 - 16, x1 + len(text) * 6.5, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), text, fill=(0, 0, 0))

    return draw_img


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get('args', {})
    num_queries = saved_args.get('num_queries', 100)
    img_size = saved_args.get('img_size', 512)
    size = (img_size, img_size)

    model = PCBDefectDetector(num_queries=num_queries, num_classes=6, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    template_t, _ = preprocess(args.template, size)
    test_t, test_orig = preprocess(args.test, size)

    with torch.no_grad():
        outputs = model(template_t.to(device), test_t.to(device))

    logits = outputs['pred_logits'][0].softmax(-1)
    boxes = outputs['pred_boxes'][0]
    scores, labels = logits[:, :-1].max(-1)
    keep = scores > args.threshold

    pred_boxes = box_cxcywh_to_xyxy(boxes[keep]).cpu().numpy()
    pred_labels = labels[keep].cpu().numpy()
    pred_scores = scores[keep].cpu().numpy()

    result = draw_boxes(test_orig, pred_boxes, pred_labels, pred_scores, size)
    result.save(args.output)

    print(f'Detected {keep.sum().item()} defect(s):')
    for label, score in zip(pred_labels, pred_scores):
        print(f'  {DEFECT_CLASSES[label]:<16}  confidence={score:.3f}')
    print(f'Saved annotated image to {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True, help='Path to defect-free template PCB image')
    parser.add_argument('--test', required=True, help='Path to manufactured (test) PCB image')
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', default='result.png', help='Output image path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    run_inference(args)
