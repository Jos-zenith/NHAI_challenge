"""Rebuild the DatalakeFace mobile inference bundles.

This helper takes the repository's exported ONNX models, applies dynamic INT8
quantization, converts the result to ORT format, and copies the generated
artifacts into the Android and iOS asset directories used by the app.

The script is intentionally conservative: it works from existing ONNX exports
already present in the repository rather than assuming the original PyTorch
architectures are available in this workspace.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.tools.convert_onnx_models_to_ort import (
    OptimizationStyle,
    convert_onnx_models_to_ort,
)


ROOT = Path(__file__).resolve().parent
APP_ASSET_DIRS = [
    ROOT / "DatalakeFace" / "android" / "app" / "src" / "main" / "assets",
    ROOT / "DatalakeFace" / "ios" / "DatalakeFace",
]

DEFAULT_FACE_ONNX = ROOT / "mobilefacenet.onnx"
DEFAULT_FACE_ORT = ROOT / "facenet_int8.ort"
DEFAULT_LIVENESS_ONNX = ROOT / "tmp_liveness.onnx"
DEFAULT_LIVENESS_ORT = ROOT / "liveness_int8.ort"


def quantize_to_ort(onnx_input_path: Path, ort_output_path: Path) -> None:
    if not onnx_input_path.exists():
        raise FileNotFoundError(f"Missing ONNX input: {onnx_input_path}")

    ort_output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        quantized_onnx_path = temp_dir / f"{onnx_input_path.stem}_int8.onnx"
        ort_output_dir = temp_dir / "ort"

        quantize_dynamic(
            model_input=str(onnx_input_path),
            model_output=str(quantized_onnx_path),
            weight_type=QuantType.QInt8,
        )

        onnx.checker.check_model(str(quantized_onnx_path))

        convert_onnx_models_to_ort(
            quantized_onnx_path,
            output_dir=ort_output_dir,
            optimization_styles=[OptimizationStyle.Fixed],
        )

        ort_candidates = sorted(ort_output_dir.rglob("*.ort"))
        if not ort_candidates:
            raise RuntimeError(
                f"ORT conversion completed but no .ort file was produced for {onnx_input_path}"
            )

        shutil.copy2(ort_candidates[0], ort_output_path)


def bundle_assets(face_ort_path: Path, liveness_ort_path: Path) -> None:
    for asset_dir in APP_ASSET_DIRS:
        asset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(face_ort_path, asset_dir / face_ort_path.name)
        shutil.copy2(liveness_ort_path, asset_dir / liveness_ort_path.name)


def build_all(
    face_onnx_path: Path = DEFAULT_FACE_ONNX,
    face_ort_path: Path = DEFAULT_FACE_ORT,
    liveness_onnx_path: Path = DEFAULT_LIVENESS_ONNX,
    liveness_ort_path: Path = DEFAULT_LIVENESS_ORT,
) -> None:
    print(f"Quantizing and converting face model from {face_onnx_path}")
    quantize_to_ort(face_onnx_path, face_ort_path)
    print(f"Wrote face ORT model to {face_ort_path}")

    print(f"Quantizing and converting liveness model from {liveness_onnx_path}")
    quantize_to_ort(liveness_onnx_path, liveness_ort_path)
    print(f"Wrote liveness ORT model to {liveness_ort_path}")

    bundle_assets(face_ort_path, liveness_ort_path)
    print("Bundled ORT models into Android and iOS asset directories")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize the repo's ONNX exports and bundle ORT assets for DatalakeFace."
    )
    parser.add_argument(
        "--face-onnx",
        type=Path,
        default=DEFAULT_FACE_ONNX,
        help="Input face embedding ONNX file.",
    )
    parser.add_argument(
        "--face-ort",
        type=Path,
        default=DEFAULT_FACE_ORT,
        help="Output face ORT file.",
    )
    parser.add_argument(
        "--liveness-onnx",
        type=Path,
        default=DEFAULT_LIVENESS_ONNX,
        help="Input liveness ONNX file.",
    )
    parser.add_argument(
        "--liveness-ort",
        type=Path,
        default=DEFAULT_LIVENESS_ORT,
        help="Output liveness ORT file.",
    )
    args = parser.parse_args()

    build_all(
        face_onnx_path=args.face_onnx,
        face_ort_path=args.face_ort,
        liveness_onnx_path=args.liveness_onnx,
        liveness_ort_path=args.liveness_ort,
    )


if __name__ == "__main__":
    main()