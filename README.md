# MobileFaceNet / Liveness export + quantize

This folder contains scripts to download MobileFaceNet, export it to ONNX, and to train a passive liveness classifier (MobileNetV2), export it to ONNX, and quantize to INT8.

Prereqs
- Python 3.8+ and a GPU if available
- Install deps:

```
pip install -r requirements.txt
```

Export MobileFaceNet to ONNX

```
python export_mobilefacenet.py --out mobilefacenet.onnx
# optional: use a local TorchScript checkpoint instead of downloading
python export_mobilefacenet.py --out mobilefacenet.onnx --checkpoint mobilefacenet_scripted.pt
```

The exporter downloads the published TorchScript checkpoint from the MobileFaceNet project and converts it to ONNX with a 1x3x112x112 input.

Train a passive liveness classifier on LCC-FASD

LCC-FASD is open and commonly used for face anti-spoofing. This script expects a
folder layout compatible with ImageFolder, with real/spoof class names under
train and val:

```
dataset/
  train/real/...
  train/spoof/...
  val/real/...
  val/spoof/...
```

Run training + export + quantize:

```
python train_liveness.py --data dataset --out-dir outputs --epochs 8 --batch-size 32 --width-mult 0.25 --image-size 160
```

Notes
- The training script fine-tunes a compact MobileNetV2 binary head for real vs spoof.
- INT8 export uses `onnxruntime.quantization.quantize_dynamic` and writes `liveness_int8.ort`.
- To stay near the 2 MB target, keep `--width-mult 0.25` and prefer the smaller `--image-size 160` or lower if your accuracy budget allows it.

Mobile app wiring
- Bundle `facenet_int8.ort` and `liveness_int8.ort` into `DatalakeFace/android/app/src/main/assets/` and `DatalakeFace/ios/DatalakeFace/`.
- The sample app loads both models through `onnxruntime-react-native` and runs them from a VisionCamera frame output.
- VisionCamera frame outputs require the worklet runtime packages, which are now installed in the React Native app.

Offline attendance storage
- The app creates an `attendances` SQLite table with `id`, `personnel_id`, `embedding`, `timestamp`, and `synced` columns.
- Live embeddings are matched against stored embeddings with cosine similarity, and values above `0.75` are treated as a match.
- When the network comes back, the NetInfo listener calls `syncAndPurge(uploadToS3)` to upload unsynced rows and delete them locally after success.
- The example `uploadToS3` callback is the integration point for your S3/Presigned-URL backend configuration.
