"""Export FaceNet (InceptionResnetV1) to ONNX and optionally quantize.

Usage:
  python export_facenet.py --out facenet.onnx
  python export_facenet.py --out facenet.onnx --quant --quant-out facenet_int8.ort
"""
import argparse
import torch

def export(out_path, opset=12):
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained='vggface2').eval()
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=['input'], output_names=['embedding'],
        opset_version=opset,
        do_constant_folding=True,
    )
    print('Exported ONNX ->', out_path)

def quantize(in_path, out_path):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        raise RuntimeError('onnxruntime with quantization support is required') from e
    quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)
    print('Quantized ONNX ->', out_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--opset', type=int, default=12)
    p.add_argument('--quant', action='store_true')
    p.add_argument('--quant-out', default='facenet_int8.ort')
    args = p.parse_args()

    export(args.out, opset=args.opset)
    if args.quant:
        quantize(args.out, args.quant_out)

if __name__ == '__main__':
    main()
