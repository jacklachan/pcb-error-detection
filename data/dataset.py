import os
import torch
from torch.utils.data import Dataset
from PIL import Image

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']


class DeepPCBDataset(Dataset):
    """
    DeepPCB dataset loader.
    Dataset: https://github.com/tangsanli5201/DeepPCB

    Directory structure:
        root/
          PCBData/
            group00041/
              00041000_temp.jpg
              00041000_test.jpg
              00041000_test.txt   (x1 y1 x2 y2 defect_type per line)
            ...
          trainval.txt  (optional split file)
          test.txt      (optional split file)
    """

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.pairs = []
        self._load_pairs(split)

    def _load_pairs(self, split):
        split_file = os.path.join(self.root, 'test.txt' if split == 'test' else 'trainval.txt')

        if os.path.exists(split_file):
            self._load_from_split_file(split_file)
        else:
            self._load_from_directory(split)

    def _load_from_split_file(self, split_file):
        pcb_data = os.path.join(self.root, 'PCBData')
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                img_path = os.path.join(pcb_data, parts[0])
                group = os.path.dirname(parts[0])
                base = os.path.splitext(os.path.basename(parts[0]))[0]
                temp_name = base.replace('_test', '_temp') + '.jpg'
                temp_path = os.path.join(pcb_data, group, temp_name)
                ann_path = os.path.join(pcb_data, parts[1]) if len(parts) > 1 else img_path.replace('.jpg', '.txt')
                if os.path.exists(img_path) and os.path.exists(temp_path):
                    self.pairs.append((temp_path, img_path, ann_path))

    def _load_from_directory(self, split):
        pcb_data = os.path.join(self.root, 'PCBData')
        if not os.path.exists(pcb_data):
            pcb_data = self.root

        groups = sorted([
            g for g in os.listdir(pcb_data)
            if os.path.isdir(os.path.join(pcb_data, g))
        ])

        n = len(groups)
        groups = groups[:int(n * 0.8)] if split == 'train' else groups[int(n * 0.8):]

        for group in groups:
            group_dir = os.path.join(pcb_data, group)
            for fname in os.listdir(group_dir):
                if not fname.endswith('_temp.jpg'):
                    continue
                base = fname.replace('_temp.jpg', '')
                test_img = os.path.join(group_dir, base + '_test.jpg')
                ann_file = os.path.join(group_dir, base + '_test.txt')
                temp_img = os.path.join(group_dir, fname)
                if os.path.exists(test_img) and os.path.exists(ann_file):
                    self.pairs.append((temp_img, test_img, ann_file))

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
                    # Normalize to [0,1] cx,cy,w,h format
                    cx = (x1 + x2) / 2 / W
                    cy = (y1 + y2) / 2 / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    boxes.append([cx, cy, bw, bh])
                    labels.append(cls - 1)  # 0-indexed

        if self.transform:
            template, test_img = self.transform(template, test_img)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }

        return template, test_img, target
