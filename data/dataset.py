import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']


class DeepPCBDataset(Dataset):
    """
    DeepPCB dataset loader — robust to all common directory layouts.
    Uses glob to find every *_temp.jpg recursively, then pairs it with
    the matching *_test.jpg and *_test.txt regardless of nesting depth.

    Dataset: https://github.com/tangsanli5201/DeepPCB
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.pairs = []
        self._load_pairs(split)
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No PCB pairs found under '{root}'.\n"
                f"Expected files matching *_temp.jpg / *_test.jpg / *_test.txt.\n"
                f"Contents of root: {os.listdir(root)}"
            )

    def _load_pairs(self, split):
        # Search for template images anywhere under root
        all_temps = sorted(glob.glob(
            os.path.join(self.root, '**', '*_temp.jpg'), recursive=True
        ))

        # Fallback: try without PCBData subdir (if root is already PCBData)
        if not all_temps:
            all_temps = sorted(glob.glob(
                os.path.join(self.root, '*_temp.jpg'), recursive=False
            ))

        valid = []
        for temp_path in all_temps:
            base = temp_path.replace('_temp.jpg', '')
            test_path = base + '_test.jpg'
            ann_path  = base + '_test.txt'
            if os.path.exists(test_path) and os.path.exists(ann_path):
                valid.append((temp_path, test_path, ann_path))

        # 80 / 20 deterministic split by group folder
        groups = sorted({os.path.dirname(p[0]) for p in valid})
        n = len(groups)
        if split == 'train':
            keep = set(groups[:int(n * 0.8)])
        else:
            keep = set(groups[int(n * 0.8):])

        self.pairs = [p for p in valid if os.path.dirname(p[0]) in keep]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        temp_path, test_path, ann_path = self.pairs[idx]

        template = Image.open(temp_path).convert('RGB')
        test_img = Image.open(test_path).convert('RGB')
        W, H = template.size

        boxes, labels = [], []
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
                labels.append(cls - 1)  # 0-indexed

        if self.transform:
            template, test_img = self.transform(template, test_img)

        return template, test_img, {
            'boxes':  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
        }
