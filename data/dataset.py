import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def _find_pairs(root):
    """
    Scan root recursively, return (template, test, annotation) triples.
    Tries four strategies from most-specific to most-general.
    """
    pairs = []

    # ── Strategy 1: standard DeepPCB — *_temp.jpg / *_test.jpg / *_test.txt ──
    for ext in ('.jpg', '.jpeg', '.png'):
        for temp in glob.glob(os.path.join(root, '**', f'*_temp{ext}'), recursive=True):
            base = temp[:-len(f'_temp{ext}')]
            for test_ext in ('.jpg', '.jpeg', '.png'):
                test = base + f'_test{test_ext}'
                ann  = base + '_test.txt'
                if os.path.exists(test) and os.path.exists(ann):
                    pairs.append((temp, test, ann))
                    break
    if pairs:
        return pairs

    # ── Strategy 2: any *_test.txt → derive image paths ─────────────────────
    for ann in glob.glob(os.path.join(root, '**', '*_test.txt'), recursive=True):
        base = ann[:-len('_test.txt')]
        test = _first_existing([base + f'_test{e}' for e in _IMG_EXTS]
                               + [base + e for e in _IMG_EXTS])
        temp = _first_existing([base + f'_temp{e}' for e in _IMG_EXTS]
                               + [base + f'_template{e}' for e in _IMG_EXTS])
        if test and temp:
            pairs.append((temp, test, ann))
    if pairs:
        return pairs

    # ── Strategy 3: trainval.txt / test.txt split files ─────────────────────
    for split_file in ['trainval.txt', 'test.txt']:
        sf = os.path.join(root, split_file)
        if not os.path.exists(sf):
            continue
        pcb_data = os.path.join(root, 'PCBData')
        if not os.path.isdir(pcb_data):
            pcb_data = root
        with open(sf) as fh:
            for line in fh:
                parts = line.strip().split()
                if not parts:
                    continue
                img_rel = parts[0]
                ann_rel = parts[1] if len(parts) > 1 else img_rel.replace('.jpg', '.txt')
                img_path = os.path.join(pcb_data, img_rel)
                ann_path = os.path.join(pcb_data, ann_rel)
                base = os.path.splitext(img_path)[0]
                temp_path = _first_existing(
                    [base.replace('_test', '_temp') + e for e in _IMG_EXTS]
                    + [os.path.splitext(img_path)[0].replace(
                        os.path.basename(img_path).split('_')[0],
                        os.path.basename(img_path).split('_')[0]) + e for e in _IMG_EXTS]
                )
                if os.path.exists(img_path) and temp_path:
                    pairs.append((temp_path, img_path, ann_path))
    if pairs:
        return pairs

    # ── Strategy 4: brute-force — any folder with ≥2 images + ≥1 .txt ──────
    # Works when naming is completely unknown (e.g. 0001.jpg, 0002.jpg, 0001.txt)
    for dirpath, _, filenames in os.walk(root):
        imgs = sorted([f for f in filenames if f.lower().endswith(_IMG_EXTS)])
        txts = sorted([f for f in filenames if f.lower().endswith('.txt')
                       and f not in ('trainval.txt', 'test.txt', 'README.txt')])
        if len(imgs) < 2 or not txts:
            continue
        # pair consecutive images: even index = template, odd index = test
        for i in range(0, len(imgs) - 1, 2):
            temp = os.path.join(dirpath, imgs[i])
            test = os.path.join(dirpath, imgs[i + 1])
            stem = os.path.splitext(imgs[i + 1])[0]
            ann_match = [os.path.join(dirpath, t) for t in txts if os.path.splitext(t)[0] == stem]
            ann = ann_match[0] if ann_match else os.path.join(dirpath, txts[0])
            pairs.append((temp, test, ann))
    return pairs


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


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
            # Print directory tree for diagnosis
            tree = []
            for dirpath, _, filenames in os.walk(root):
                depth = dirpath.replace(root, '').count(os.sep)
                if depth > 3:
                    continue
                tree.append(f"{'  '*depth}{os.path.basename(dirpath)}/")
                for f in sorted(filenames)[:8]:
                    tree.append(f"{'  '*(depth+1)}{f}")
            raise RuntimeError(
                f"No PCB pairs found under '{root}'.\n"
                "Check that images were downloaded (not LFS pointers).\n"
                + '\n'.join(tree)
            )

        # 80 / 20 split by parent folder
        groups = sorted({os.path.dirname(p[0]) for p in all_pairs})
        n = len(groups)
        n_train = max(1, int(n * 0.8))
        n_val   = max(1, n - n_train)
        if n == 1:
            keep = set(groups)
        elif split == 'train':
            keep = set(groups[:n_train])
        else:
            keep = set(groups[n - n_val:])
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
                    x1, y1, x2, y2, cls = (int(parts[k]) for k in range(5))
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
