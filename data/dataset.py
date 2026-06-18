import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']


def _find_pairs(root):
    """
    Scan root recursively and return all valid (template, test, annotation) triples.
    Tries every known DeepPCB naming convention.
    """
    pairs = []

    # Strategy 1 — standard DeepPCB naming: *_temp.jpg + *_test.jpg + *_test.txt
    for temp in glob.glob(os.path.join(root, '**', '*_temp.jpg'), recursive=True):
        base = temp[:-len('_temp.jpg')]
        test = base + '_test.jpg'
        ann  = base + '_test.txt'
        if os.path.exists(test) and os.path.exists(ann):
            pairs.append((temp, test, ann))

    if pairs:
        return pairs

    # Strategy 2 — annotation txt decides: find every *_test.txt, look for matching images
    for ann in glob.glob(os.path.join(root, '**', '*_test.txt'), recursive=True):
        base = ann[:-len('_test.txt')]
        test = base + '_test.jpg'
        temp = base + '_temp.jpg'
        if not os.path.exists(test):
            test = base + '.jpg'           # some variants drop the _test suffix on image
        if not os.path.exists(temp):
            temp = base + '_template.jpg'
        if os.path.exists(test) and os.path.exists(temp):
            pairs.append((temp, test, ann))

    if pairs:
        return pairs

    # Strategy 3 — trainval.txt / test.txt split files
    for split_file in ['trainval.txt', 'test.txt']:
        sf = os.path.join(root, split_file)
        if not os.path.exists(sf):
            continue
        pcb_data = os.path.join(root, 'PCBData')
        if not os.path.isdir(pcb_data):
            pcb_data = root
        with open(sf) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_rel = parts[0]
                ann_rel = parts[1] if len(parts) > 1 else img_rel.replace('.jpg', '.txt')
                img_path = os.path.join(pcb_data, img_rel)
                ann_path = os.path.join(pcb_data, ann_rel)
                base = os.path.splitext(img_path)[0].replace('_test', '_temp')
                temp_path = base + '.jpg'
                if os.path.exists(img_path) and os.path.exists(temp_path):
                    pairs.append((temp_path, img_path, ann_path))

    return pairs


class DeepPCBDataset(Dataset):
    """
    DeepPCB dataset — auto-detects naming convention.
    Dataset: https://github.com/tangsanli5201/DeepPCB
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        all_pairs = _find_pairs(root)

        if not all_pairs:
            # Print full tree for diagnosis
            tree = []
            for dirpath, _, filenames in os.walk(root):
                depth = dirpath.replace(root, '').count(os.sep)
                if depth > 3:
                    continue
                tree.append(f"{'  '*depth}{os.path.basename(dirpath)}/")
                for f in filenames[:5]:
                    tree.append(f"{'  '*(depth+1)}{f}")
            raise RuntimeError(
                f"No PCB pairs found under '{root}'.\n"
                + '\n'.join(tree)
            )

        # 80 / 20 deterministic split by parent folder
        groups = sorted({os.path.dirname(p[0]) for p in all_pairs})
        n = len(groups)
        keep = set(groups[:int(n * 0.8)]) if split == 'train' else set(groups[int(n * 0.8):])
        self.pairs = [p for p in all_pairs if os.path.dirname(p[0]) in keep]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        temp_path, test_path, ann_path = self.pairs[idx]
        template = Image.open(temp_path).convert('RGB')
        test_img = Image.open(test_path).convert('RGB')
        W, H = template.size

        boxes, labels = [], []
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    x1, y1, x2, y2, cls = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    cx = (x1 + x2) / 2 / W
                    cy = (y1 + y2) / 2 / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    boxes.append([cx, cy, bw, bh])
                    labels.append(cls - 1)

        if self.transform:
            template, test_img = self.transform(template, test_img)

        return template, test_img, {
            'boxes':  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
        }
