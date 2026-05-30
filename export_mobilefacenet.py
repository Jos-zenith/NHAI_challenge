"""Download MobileFaceNet and export it to ONNX.

Usage:
  python export_mobilefacenet.py --out mobilefacenet.onnx
  python export_mobilefacenet.py --out mobilefacenet.onnx --checkpoint mobilefacenet_scripted.pt
  python export_mobilefacenet.py --out mobilefacenet.onnx --checkpoint-url <url>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen

import torch


DEFAULT_CHECKPOINT_URL = (
    "https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/"
    "mobilefacenet_scripted.pt"
)


def download_checkpoint(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    print(f"Downloading MobileFaceNet checkpoint from {url}")
    with urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)

    return destination


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    try:
        model = torch.jit.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TorchScript checkpoint: {checkpoint_path}"
        ) from exc

    model.eval()
    return model


def export(out_path: Path, checkpoint_path: Path, opset: int = 12) -> None:
    model = load_model(checkpoint_path)
    dummy = torch.randn(1, 3, 112, 112)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output ONNX file path")
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version to export with"
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Local TorchScript checkpoint path. If omitted, the default checkpoint is downloaded.",
    )
    parser.add_argument(
        "--checkpoint-url",
        default=DEFAULT_CHECKPOINT_URL,
        help="TorchScript checkpoint URL to download when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--download-dir",
        default="weights",
        help="Directory used to cache downloaded checkpoints.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        filename = Path(args.checkpoint_url).name or "mobilefacenet_scripted.pt"
        checkpoint_path = download_checkpoint(
            args.checkpoint_url, Path(args.download_dir) / filename
        )

    export(out_path, checkpoint_path, opset=args.opset)


if __name__ == "__main__":
    main()