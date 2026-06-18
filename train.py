"""
Training script for PCB Defect Detector.

Usage:
    python train.py --data_root path/to/DeepPCB
    python train.py --data_root path/to/DeepPCB --backbone swin_t --epochs 100 --amp
    python train.py --data_root path/to/DeepPCB --resume output/checkpoint_epoch50.pth
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import DeepPCBDataset
from data.transforms import PCBTransform
from models.pcb_detector import PCBDefectDetector
from utils.losses import PCBDetectionLoss
from utils.visualize import plot_training_curves


def collate_fn(batch):
    templates, tests, targets = zip(*batch)
    return torch.stack(templates), torch.stack(tests), list(targets)


def build_loaders(args):
    train_ds = DeepPCBDataset(
        args.data_root, split='train',
        transform=PCBTransform((args.img_size, args.img_size), train=True),
    )
    val_ds = DeepPCBDataset(
        args.data_root, split='test',
        transform=PCBTransform((args.img_size, args.img_size), train=False),
    )
    print(f'Train: {len(train_ds)} pairs  |  Val: {len(val_ds)} pairs')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True, use_amp=False):
    model.train(train)
    totals = {'loss_ce': 0.0, 'loss_bbox': 0.0, 'loss_giou': 0.0, 'total': 0.0}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for templates, tests, targets in tqdm(loader, leave=False):
            templates = templates.to(device, non_blocking=True)
            tests     = tests.to(device, non_blocking=True)
            targets   = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(enabled=use_amp):
                outputs = model(templates, tests)
                losses  = criterion(outputs, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(losses['total']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.amp and device.type == 'cuda'
    print(f'Device: {device}  |  AMP: {use_amp}  |  Backbone: {args.backbone}')

    train_loader, val_loader = build_loaders(args)

    model = PCBDefectDetector(
        backbone=args.backbone,
        num_queries=args.num_queries,
        num_classes=6,
        pretrained=True,
        num_decoder_layers=args.decoder_layers,
    ).to(device)

    criterion = PCBDetectionLoss(num_classes=6).to(device)

    # Backbone uses 10x lower LR — standard practice when fine-tuning pretrained weights
    param_dicts = [
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.1},
    ]
    optimizer  = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler     = GradScaler(enabled=use_amp)

    start_epoch = 1
    train_losses, val_losses = [], []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        train_losses = ckpt.get('train_losses', [])
        val_losses   = ckpt.get('val_losses', [])
        print(f'Resumed from epoch {ckpt["epoch"]}')

    best_val = min(val_losses) if val_losses else float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer,
                            scaler, device, train=True,  use_amp=use_amp)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer,
                            scaler, device, train=False, use_amp=use_amp)
        scheduler.step()

        train_losses.append(train_m['total'])
        val_losses.append(val_m['total'])

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"Train: {train_m['total']:.4f} "
            f"(ce={train_m['loss_ce']:.3f} "
            f"bbox={train_m['loss_bbox']:.3f} "
            f"giou={train_m['loss_giou']:.3f})  |  "
            f"Val: {val_m['total']:.4f}"
        )

        if val_m['total'] < best_val:
            best_val = val_m['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  ✓ Best model saved  (val_loss={best_val:.4f})')

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': vars(args),
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))

    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(args.output_dir, 'training_curves.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     type=str,   required=True)
    parser.add_argument('--output_dir',    type=str,   default='output')
    parser.add_argument('--backbone',      type=str,   default='resnet50',
                        choices=['resnet50', 'swin_t'])
    parser.add_argument('--img_size',      type=int,   default=512)
    parser.add_argument('--batch_size',    type=int,   default=4)
    parser.add_argument('--num_queries',   type=int,   default=100)
    parser.add_argument('--decoder_layers',type=int,   default=6)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--resume',        type=str,   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--amp',           action='store_true',
                        help='Use automatic mixed precision (CUDA only)')
    args = parser.parse_args()
    main(args)
