import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

DEFECT_CLASSES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']


def _find_pairs(root):
    """
    Return (template_path, test_path, annotation_path) triples from the DeepPCB repo.

    Actual repo structure (confirmed by inspection):
        root/
          PCBData/
            trainval.txt     <- "groupXX/XX/XXX.jpg  groupXX/XX_not/XXX.txt"
            test.txt
            groupXXXXX/
              XXXXX/         <- test images + template images
                XXXXXXXX.jpg          (test image, NO _test suffix)
                XXXXXXXX_temp.jpg     (template image)
              XXXXX_not/     <- annotation txts
                XXXXXXXX.txt
    """
    pairs = []

    # Locate PCBData sub-directory (or use root if already there)
    pcb_data = os.path.join(root, 'PCBData')
    if not os.path.isdir(pcb_data):
        pcb_data = root

    # ── Strategy 1: use official split files (trainval.txt / test.txt) ────────
    for split_name in ('trainval.txt', 'test.txt'):
        sf = os.path.join(pcb_data, split_name)
        if not os.path.exists(sf):
            continue
        with open(sf) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                test_rel, ann_rel = parts[0], parts[1]
                test_path = os.path.join(pcb_data, test_rel)
                ann_path  = os.path.join(pcb_data, ann_rel)
                # Template = same directory, same basename + _temp.jpg
                stem = os.path.splitext(test_rel)[0]   # e.g. group20085/20085/20085000
                temp_path = os.path.join(pcb_data, stem + '_temp.jpg')
                if os.path.exists(test_path) and os.path.exists(temp_path):
                    pairs.append((temp_path, test_path, ann_path))

    if pairs:
        return pairs

    # ── Strategy 2: scan for *_temp.jpg, infer test + annotation paths ────────
    # Handles cases where trainval.txt is absent but directory layout is intact.
    for temp in glob.glob(os.path.join(root, '**', '*_temp.jpg'), recursive=True):
        stem = temp[:-len('_temp.jpg')]   # e.g. .../20085/20085000
        test = stem + '.jpg'              # test image has NO suffix
        if not os.path.exists(test):
            test = stem + '_test.jpg'     # older convention
        if not os.path.exists(test):
            continue

        # Annotation lives in a sibling folder named <subdir>_not/
        # e.g.  .../group20085/20085/20085000_temp.jpg
        #        → .../group20085/20085_not/20085000.txt
        subdir   = os.path.dirname(stem)           # .../group20085/20085
        basename = os.path.basename(stem)          # 20085000
        parent   = os.path.dirname(subdir)         # .../group20085
        not_dir  = os.path.join(parent, os.path.basename(subdir) + '_not')
        ann = os.path.join(not_dir, basename + '.txt')
        if not os.path.exists(ann):
            ann = stem + '.txt'           # same folder fallback
        if not os.path.exists(ann):
            ann = stem + '_test.txt'

        pairs.append((temp, test, ann))

    return pairs


class DeepPCBDataset(Dataset):
    """DeepPCB dataset. https://github.com/tangsanli5201/DeepPCB"""

    def __init__(self, root, split='train', transform=None):
        self.transform = transform

        all_pairs = _find_pairs(root)

        if not all_pairs:
            # Print directory tree so the caller can diagnose
            lines = []
            for dp, _, fnames in os.walk(root):
                depth = dp.replace(root, '').count(os.sep)
                if depth > 5:
                    continue
                lines.append('  ' * depth + os.path.basename(dp) + '/')
                for fn in sorted(fnames)[:6]:
                    sz = os.path.getsize(os.path.join(dp, fn))
                    lines.append('  ' * (depth + 1) + f'{fn}  ({sz} B)')
            raise RuntimeError(
                f"No pairs found under '{root}'.\n" + '\n'.join(lines)
            )

        # Split 80 / 20 by top-level group folder (two levels up from image file)
        def _group_key(path):
            # path: .../PCBData/group20085/20085/20085000_temp.jpg
            # key : .../PCBData/group20085
            return os.path.dirname(os.path.dirname(path))

        groups = sorted({_group_key(p[0]) for p in all_pairs})
        n = len(groups)
        n_train = max(1, int(n * 0.8))
        n_val   = max(1, n - n_train)

        if n == 1:
            keep = set(groups)
        elif split == 'train':
            keep = set(groups[:n_train])
        else:
            keep = set(groups[n - n_val:])

        self.pairs = [p for p in all_pairs if _group_key(p[0]) in keep]

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
                    x1, y1, x2, y2, cls = (int(p) for p in parts[:5])
                    boxes.append([
                        (x1 + x2) / 2 / W,
                        (y1 + y2) / 2 / H,
                        (x2 - x1) / W,
                        (y2 - y1) / H,
                    ])
                    labels.append(cls - 1)   # 1-indexed → 0-indexed

        if self.transform:
            template, test_img = self.transform(template, test_img)

        return template, test_img, {
            'boxes':  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
        }
