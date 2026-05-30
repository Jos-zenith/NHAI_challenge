"""Train a passive liveness classifier on an LCC-FASD-style dataset.

Expected layout:
  dataset/
    train/real/...
    train/spoof/...
    val/real/...
    val/spoof/...

This script fine-tunes a compact MobileNetV2, exports ONNX, and applies
onnxruntime dynamic INT8 quantization in the same run.

Usage:
  python train_liveness.py --data dataset --out-dir outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


REAL_CLASS_NAMES = {"real", "live", "bonafide"}
SPOOF_CLASS_NAMES = {"spoof", "fake", "attack", "print", "replay"}


class LivenessFolder(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        path = self.dataset.samples[index][0]
        class_name = Path(path).parent.name.lower()
        if class_name in REAL_CLASS_NAMES:
            label = 1
        elif class_name in SPOOF_CLASS_NAMES:
            label = 0
        else:
            label = int(self.dataset.targets[index])
        return image, label


def build_model(pretrained=True, width_mult=0.25):
    if pretrained:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights, width_mult=width_mult)
    else:
        model = models.mobilenet_v2(weights=None, width_mult=width_mult)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def build_transform(image_size=160):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_split_root(data_dir, split_name):
    split_root = os.path.join(data_dir, split_name)
    if os.path.isdir(split_root):
        return split_root
    return data_dir


def export_onnx(model, out_path, image_size, opset=12):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=['input'],
        output_names=['logit'],
        dynamic_axes={'input': {0: 'batch'}, 'logit': {0: 'batch'}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print('Exported ONNX ->', out_path)


def quantize_onnx(onnx_path, quant_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
    print('Quantized INT8 ->', quant_path)


def train(data_dir, out_dir, epochs=5, batch_size=32, lr=1e-3, width_mult=0.25, image_size=160):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = build_transform(image_size=image_size)

    train_root = get_split_root(data_dir, 'train')
    val_root = get_split_root(data_dir, 'val')

    train_ds = LivenessFolder(train_root, transform=transform)
    val_ds = LivenessFolder(val_root, transform=transform)

    num_workers = 4 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(pretrained=True, width_mult=width_mult).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        running_loss /= len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().squeeze(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total else 0.0
        print(f'Epoch {epoch}: loss={running_loss:.4f} val_acc={acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_dir / 'liveness_mobilenetv2.pth')

    best_path = out_dir / 'liveness_mobilenetv2.pth'
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    onnx_path = out_dir / 'liveness.onnx'
    quant_path = out_dir / 'liveness_int8.ort'

    export_onnx(model, onnx_path, image_size=image_size)
    try:
        quantize_onnx(onnx_path, quant_path)
        print('Quantized file size:', quant_path.stat().st_size, 'bytes')
    except Exception as exc:
        print('onnxruntime quantization not available — skip quantization')
        print(exc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Dataset root containing train/val or a flat LCC-FASD ImageFolder tree')
    p.add_argument('--out-dir', default='outputs')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--width-mult', type=float, default=0.25, help='MobileNetV2 width multiplier; 0.25 is small enough to help INT8 stay under 2 MB')
    p.add_argument('--image-size', type=int, default=160, help='Training and export input size')
    args = p.parse_args()
    train(
        args.data,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        width_mult=args.width_mult,
        image_size=args.image_size,
    )


if __name__ == '__main__':
    main()
