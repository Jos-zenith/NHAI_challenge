import { useEffect, useRef } from 'react';
import {
  ActivityIndicator,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import NetInfo from '@react-native-community/netinfo';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import * as ort from 'onnxruntime-react-native';
import { scheduleOnRN } from 'react-native-worklets';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import {
  authenticateEmbedding,
  initAttendanceDb,
  getUnsyncedCount,
  saveAttendance,
  syncAndPurge,
} from './src/attendanceStore';
import { syncEmployeePull } from './src/employeeSync';
import { pushLocalChangesToServer } from './src/pushSync';
import { useBiometricUIState } from './src/useBiometricUIState';

const FACE_MODEL = 'facenet_int8.ort';
const FACE_INPUT_SIZE = 112;
const LIVENESS_MODEL = 'liveness_int8.ort';
const LIVENESS_INPUT_SIZE = 160;
const LIVENESS_THRESHOLD = 0.9;

function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" />
      <AppContent />
    </SafeAreaProvider>
  );
}

function AppContent() {
  const insets = useSafeAreaInsets();
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const faceSessionRef = useRef<ort.InferenceSession | null>(null);
  const livenessSessionRef = useRef<ort.InferenceSession | null>(null);
  const { uiState, transitionTo } = useBiometricUIState();

  const statusText = (() => {
    switch (uiState.progressState) {
      case 'INITIALIZING':
        return 'loading models and preparing biometric state';
      case 'READY':
        return 'models loaded and attendance DB ready';
      case 'LIVENESS_RUNNING':
        return `checking liveness${uiState.activeChallenge ? `: ${uiState.activeChallenge}` : ''}`;
      case 'EMBEDDING_RUNNING':
        return 'running embedding match';
      case 'AUTH_SUCCESS':
        return uiState.similarityScore !== null
          ? `match confirmed at ${uiState.similarityScore.toFixed(3)}`
          : 'match confirmed';
      case 'AUTH_FAILED':
        return uiState.errorMessage ?? 'authentication failed';
      case 'SYNCING':
        return 'syncing offline rows and remote changes';
      default:
        return 'ready';
    }
  })();

  const syncText = uiState.unsyncedRowsCount > 0
    ? `${uiState.unsyncedRowsCount} unsynced row${uiState.unsyncedRowsCount === 1 ? '' : 's'} pending`
    : 'no unsynced rows';

  const authText = uiState.errorMessage ?? (uiState.similarityScore !== null
    ? `similarity ${uiState.similarityScore.toFixed(3)}`
    : 'awaiting a valid embedding');

  const lastResult = 'inference results are logged from the frame processor';

  const handleEmbeddingOnJs = async (embeddingValues: number[]) => {
    if (uiState.progressState !== 'EMBEDDING_RUNNING') {
      transitionTo('EMBEDDING_RUNNING', { errorMessage: null });
    }

    const embedding = new Float32Array(embeddingValues);
    const match = await authenticateEmbedding(embedding);

    if (!match) {
      transitionTo('AUTH_FAILED', {
        errorMessage: 'no cosine match above 0.75',
        similarityScore: null,
        activeChallenge: null,
      });
      return;
    }

    transitionTo('AUTH_SUCCESS', {
      similarityScore: match.similarity,
      errorMessage: null,
      activeChallenge: null,
    });
    await saveAttendance({
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      personnel_id: match.personnelId,
      embedding,
      timestamp: Date.now(),
      synced: 0,
    });
    transitionTo('AUTH_SUCCESS', {
      unsyncedRowsCount: await getUnsyncedCount(),
      similarityScore: match.similarity,
      errorMessage: null,
      activeChallenge: null,
    });
  };

  useEffect(() => {
    let isMounted = true;

    async function bootstrap() {
      try {
        transitionTo('INITIALIZING', { errorMessage: null });
        await initAttendanceDb();
        const faceSession = await ort.InferenceSession.create(FACE_MODEL);
        const livenessSession = await ort.InferenceSession.create(LIVENESS_MODEL);

        if (!isMounted) {
          return;
        }

        faceSessionRef.current = faceSession;
        livenessSessionRef.current = livenessSession;
        transitionTo('READY', { unsyncedRowsCount: await getUnsyncedCount() });
      } catch (error) {
        if (isMounted) {
          transitionTo('AUTH_FAILED', { errorMessage: `model load failed: ${String(error)}` });
        }
      }
    }

    void bootstrap();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener(state => {
      if (state.isConnected) {
        transitionTo('SYNCING', {
          errorMessage: null,
          unsyncedRowsCount: uiState.unsyncedRowsCount,
        });
        void syncAndPurge(uploadToS3)
          .then(() => pushLocalChangesToServer())
          .then(() => syncEmployeePull())
          .then(async () => {
            transitionTo('READY', {
              unsyncedRowsCount: await getUnsyncedCount(),
              errorMessage: null,
            });
          })
          .catch(error =>
            transitionTo('AUTH_FAILED', {
              errorMessage: `sync failed: ${String(error)}`,
            }),
          );
      }
    });

    return unsubscribe;
  }, []);

  useEffect(() => {
    if (!hasPermission) {
      void requestPermission();
    }
  }, [hasPermission, requestPermission]);

  // Shim: provide a `useFrameProcessor`-like hook on top of the existing
  // `useFrameOutput` API so callers can use the newer hook shape while
  // remaining compatible with the installed `react-native-vision-camera` v5.
  function useFrameProcessorShim(processor: (frame: any) => void, deps: any[]) {
    const frameOutput = useFrameOutput({
      onFrame(frame) {
        'worklet';
        processor(frame);
      },
    });

    return frameOutput;
  }

  const frameProcessor = useFrameProcessorShim((frame) => {
    'worklet';

    const faceSession = faceSessionRef.current;
    const livenessSession = livenessSessionRef.current;

    if (!faceSession || !livenessSession) {
      frame.dispose();
      return;
    }

    if (uiState.progressState !== 'LIVENESS_RUNNING') {
      transitionTo('LIVENESS_RUNNING', { activeChallenge: 'BLINK', errorMessage: null });
    }

    // schedule the heavy JS inference on the RN JS thread
    scheduleOnRN(processFrame, frame, faceSession, livenessSession, handleEmbeddingOnJs);
  }, []);

  if (!device) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator color="#fff" />
        <Text style={styles.statusText}>Waiting for camera device</Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.title}>Camera permission required</Text>
        <Text style={styles.body}>
          Grant camera access to run face embedding and liveness inference in the frame processor.
        </Text>
        <Text style={styles.statusText}>{statusText}</Text>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        outputs={[frameProcessor]}
      />

      <View style={[styles.overlay, { paddingTop: insets.top + 16 }]}> 
        <Text style={styles.kicker}>MobileFaceNet + Liveness</Text>
        <Text style={styles.title}>Offline attendance capture</Text>
        <Text style={styles.body}>{statusText}</Text>
        <Text style={styles.body}>{syncText}</Text>
        <Text style={styles.body}>{authText}</Text>
        <Text style={styles.body}>Progress state: {uiState.progressState}</Text>
        <Text style={styles.body}>{lastResult}</Text>
        <Text style={styles.caption}>
          Face embedding model: {FACE_MODEL}
        </Text>
        <Text style={styles.caption}>SQLite table: attendances</Text>
      </View>
    </View>
  );
}

async function processFrame(
  frame: any,
  faceSession: ort.InferenceSession,
  livenessSession: ort.InferenceSession,
  onEmbeddingMatched: (embeddingValues: number[]) => Promise<void> | void,
) {
  try {
    const faceBox = resolveFaceBoundingBox(frame);
    const livenessInput = frameToInputTensor(frame, LIVENESS_INPUT_SIZE, 'liveness', faceBox);
    const livenessResult = await livenessSession.run({ input: livenessInput });
    const livenessProbability = extractLivenessProbability(livenessResult);

    if (livenessProbability === null) {
      console.log('liveness probability missing from liveness model output');
      return;
    }

    if (livenessProbability < LIVENESS_THRESHOLD) {
      console.log(`liveness rejected: ${livenessProbability.toFixed(3)}`);
      return;
    }

    const input = frameToInputTensor(frame, FACE_INPUT_SIZE, 'face', faceBox);
    const faceResult = await faceSession.run({ input });
    const embedding = extractEmbedding(faceResult);

    if (!embedding) {
      console.log('embedding missing from face model output');
      return;
    }

    scheduleOnRN(onEmbeddingMatched, Array.from(embedding));
  } catch (error) {
    console.log(`inference error: ${String(error)}`);
  } finally {
    frame.dispose();
  }
}

function extractEmbedding(result: ort.InferenceSession.OnnxValueMapType) {
  const output = result.embedding ?? result['embedding'];
  if (!output || !('data' in output)) {
    return null;
  }

  const data = output.data;
  if (data instanceof Float32Array) {
    return data;
  }

  if (ArrayBuffer.isView(data)) {
    return new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  }

  return null;
}

function extractLivenessProbability(result: ort.InferenceSession.OnnxValueMapType) {
  const output =
    result.logits ??
    result.logit ??
    result.output ??
    result['logits'] ??
    result['logit'] ??
    result['output'];
  if (!output || !('data' in output)) {
    return null;
  }

  const data = output.data;
  const values =
    data instanceof Float32Array
      ? data
      : ArrayBuffer.isView(data)
        ? new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
        : null;

  if (!values || values.length === 0) {
    return null;
  }

  if (values.length >= 2) {
    const spoofLogit = values[0];
    const realLogit = values[1];
    const maxLogit = Math.max(spoofLogit, realLogit);
    const spoofExp = Math.exp(spoofLogit - maxLogit);
    const realExp = Math.exp(realLogit - maxLogit);
    return realExp / (spoofExp + realExp);
  }

  const singleLogit = values[0];
  return 1 / (1 + Math.exp(-singleLogit));
}

type PreprocessMode = 'face' | 'liveness';

type FaceBoundingBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

function frameToInputTensor(
  frame: any,
  size: number,
  mode: PreprocessMode,
  faceBox: FaceBoundingBox | null,
) {
  const sourceWidth = Number(frame.width ?? frame.imageWidth ?? frame.formatWidth ?? 0);
  const sourceHeight = Number(frame.height ?? frame.imageHeight ?? frame.formatHeight ?? 0);

  if (!sourceWidth || !sourceHeight) {
    throw new Error('frame dimensions are unavailable for preprocessing');
  }

  const cropRegion = faceBox
    ? expandBoundingBox(faceBox, sourceWidth, sourceHeight, 1.2)
    : createCenteredSquareRegion(sourceWidth, sourceHeight);
  const cropSize = cropRegion.side;
  const cropLeft = cropRegion.left;
  const cropTop = cropRegion.top;
  const chw = new Float32Array(3 * size * size);
  const pixelCount = size * size;
  const pixelReader = createRgbReader(frame);

  for (let y = 0; y < size; y += 1) {
    const canvasY = cropTop + Math.min(cropSize - 1, Math.floor((y * cropSize) / size));

    for (let x = 0; x < size; x += 1) {
      const canvasX = cropLeft + Math.min(cropSize - 1, Math.floor((x * cropSize) / size));
      const destinationIndex = y * size + x;

      const isInsideCrop =
        canvasX >= cropRegion.boxLeft &&
        canvasX < cropRegion.boxLeft + cropRegion.boxWidth &&
        canvasY >= cropRegion.boxTop &&
        canvasY < cropRegion.boxTop + cropRegion.boxHeight;
      const isInsideFrame = canvasX >= 0 && canvasX < sourceWidth && canvasY >= 0 && canvasY < sourceHeight;
      const samplePixel = isInsideCrop && isInsideFrame;
      const rgb = samplePixel ? pixelReader(canvasX, canvasY) : GRAY_RGB;

      chw[destinationIndex] = normalizeChannel(rgb.r, mode);
      chw[pixelCount + destinationIndex] = normalizeChannel(rgb.g, mode);
      chw[pixelCount * 2 + destinationIndex] = normalizeChannel(rgb.b, mode);
    }
  }

  return new ort.Tensor('float32', chw, [1, 3, size, size]);
}

function normalizeChannel(value: number, mode: PreprocessMode) {
  if (mode === 'face') {
    return (value - 0.5) / 0.5;
  }

  return value;
}

const GRAY_RGB = { r: 128 / 255, g: 128 / 255, b: 128 / 255 };

function createRgbReader(frame: any) {
  if (frame.isPlanar || String(frame.pixelFormat ?? '').toLowerCase().startsWith('yuv')) {
    return createYuvRgbReader(frame);
  }

  return createPackedRgbReader(frame);
}

function createPackedRgbReader(frame: any) {
  const pixelBuffer = frame.getPixelBuffer();
  const pixels = new Uint8Array(pixelBuffer);
  const bytesPerRow = Number(frame.bytesPerRow) || frame.width * 4;
  const pixelFormat = String(frame.pixelFormat ?? '').toLowerCase();
  const redIndex = pixelFormat.includes('rgba') ? 0 : 2;
  const blueIndex = pixelFormat.includes('rgba') ? 2 : 0;

  return (x: number, y: number) => {
    const offset = y * bytesPerRow + x * 4;
    return {
      r: pixels[offset + redIndex] / 255,
      g: pixels[offset + 1] / 255,
      b: pixels[offset + blueIndex] / 255,
    };
  };
}

function createYuvRgbReader(frame: any) {
  const planes = frame.getPlanes?.() ?? [];
  if (planes.length < 2) {
    return createPackedRgbReader(frame);
  }

  const yPlane = new Uint8Array(planes[0].getPixelBuffer());
  const yBytesPerRow = Number(planes[0].bytesPerRow ?? frame.width ?? 0) || frame.width;
  const uvPlane = new Uint8Array(planes[1].getPixelBuffer());
  const uvBytesPerRow = Number(planes[1].bytesPerRow) || Math.ceil(frame.width / 2) * 2;
  const hasSeparateUVPlanes = planes.length >= 3;

  let uPlane: Uint8Array | null = null;
  let vPlane: Uint8Array | null = null;

  if (hasSeparateUVPlanes) {
    uPlane = uvPlane;
    vPlane = new Uint8Array(planes[2].getPixelBuffer());
  }

  return (x: number, y: number) => {
    const yIndex = y * yBytesPerRow + x;
    const luma = yPlane[yIndex] / 255;
    const chromaX = Math.floor(x / 2);
    const chromaY = Math.floor(y / 2);

    let u: number;
    let v: number;

    if (hasSeparateUVPlanes && uPlane && vPlane) {
      const uIndex = chromaY * Number(planes[1].bytesPerRow ?? Math.ceil(frame.width / 2)) + chromaX;
      const vIndex = chromaY * Number(planes[2].bytesPerRow ?? Math.ceil(frame.width / 2)) + chromaX;
      u = uPlane[uIndex] / 255;
      v = vPlane[vIndex] / 255;
    } else {
      const uvIndex = chromaY * uvBytesPerRow + chromaX * 2;
      u = uvPlane[uvIndex] / 255;
      v = uvPlane[uvIndex + 1] / 255;
    }

    const yValue = luma * 255;
    const uValue = u * 255 - 128;
    const vValue = v * 255 - 128;

    const r = clamp01((yValue + 1.402 * vValue) / 255);
    const g = clamp01((yValue - 0.344136 * uValue - 0.714136 * vValue) / 255);
    const b = clamp01((yValue + 1.772 * uValue) / 255);

    return { r, g, b };
  };
}

function clamp01(value: number) {
  if (value < 0) {
    return 0;
  }

  if (value > 1) {
    return 1;
  }

  return value;
}

function resolveFaceBoundingBox(frame: any): FaceBoundingBox | null {
  const candidate =
    frame.faceBoundingBox ??
    frame.faceBox ??
    frame.boundingBox ??
    frame.detectorBox ??
    frame.detectedFaceBox;

  if (!candidate) {
    return null;
  }

  const x = Number(candidate.x ?? candidate.left ?? candidate.originX);
  const y = Number(candidate.y ?? candidate.top ?? candidate.originY);
  const w = Number(candidate.w ?? candidate.width);
  const h = Number(candidate.h ?? candidate.height);

  if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) {
    return null;
  }

  return { x, y, w, h };
}

function createCenteredSquareRegion(sourceWidth: number, sourceHeight: number) {
  const side = Math.min(sourceWidth, sourceHeight);
  const left = Math.floor((sourceWidth - side) / 2);
  const top = Math.floor((sourceHeight - side) / 2);

  return {
    left,
    top,
    side,
    boxLeft: left,
    boxTop: top,
    boxWidth: side,
    boxHeight: side,
  };
}

function expandBoundingBox(
  box: FaceBoundingBox,
  sourceWidth: number,
  sourceHeight: number,
  scale: number,
) {
  const centerX = box.x + box.w / 2;
  const centerY = box.y + box.h / 2;
  const expandedWidth = box.w * scale;
  const expandedHeight = box.h * scale;
  const left = Math.floor(centerX - expandedWidth / 2);
  const top = Math.floor(centerY - expandedHeight / 2);
  const side = Math.ceil(Math.max(expandedWidth, expandedHeight));

  return {
    left,
    top,
    side,
    boxLeft: Math.floor(centerX - expandedWidth / 2),
    boxTop: Math.floor(centerY - expandedHeight / 2),
    boxWidth: Math.ceil(expandedWidth),
    boxHeight: Math.ceil(expandedHeight),
  };
}

function isBgraFrame(frame: any) {
  const pixelFormat = String(frame.pixelFormat ?? frame.format ?? '').toUpperCase();
  return pixelFormat.includes('BGRA');
}

async function uploadToS3(row: {
  id: string;
  personnel_id: string;
  embedding: Float32Array;
  timestamp: number;
  synced: number;
}) {
  throw new Error(`Configure uploadToS3 for row ${row.id} before enabling sync`);
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: '#050505',
  },
  overlay: {
    flex: 1,
    justifyContent: 'flex-start',
    paddingHorizontal: 20,
    paddingBottom: 24,
    backgroundColor: 'rgba(5, 5, 5, 0.18)',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#050505',
    paddingHorizontal: 24,
  },
  kicker: {
    color: '#8dd9ff',
    textTransform: 'uppercase',
    letterSpacing: 1.2,
    fontSize: 12,
    marginBottom: 10,
  },
  title: {
    color: '#ffffff',
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 12,
  },
  body: {
    color: '#d9e3ea',
    fontSize: 16,
    lineHeight: 22,
    marginBottom: 8,
  },
  caption: {
    color: '#9aa7b2',
    fontSize: 13,
    marginTop: 4,
  },
  statusText: {
    color: '#d9e3ea',
    fontSize: 16,
    marginTop: 12,
    textAlign: 'center',
  },
});

export default App;
