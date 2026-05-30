import { useEffect, useRef, useState } from 'react';
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
  saveAttendance,
  syncAndPurge,
} from './src/attendanceStore';

const FACE_MODEL = 'facenet_int8.ort';
const FACE_INPUT_SIZE = 112;
const LIVENESS_MODEL = 'liveness_int8.ort';
const LIVENESS_INPUT_SIZE = 160;
const LIVENESS_THRESHOLD = 0.5;

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
  const [modelStatus, setModelStatus] = useState('loading models');
  const [syncStatus, setSyncStatus] = useState('sync idle');
  const [authStatus, setAuthStatus] = useState('awaiting a valid embedding');
  const [lastResult] = useState('inference results are logged from the frame processor');

  const handleEmbeddingOnJs = async (embeddingValues: number[]) => {
    const embedding = new Float32Array(embeddingValues);
    const match = await authenticateEmbedding(embedding);

    if (!match) {
      setAuthStatus('no cosine match above 0.75');
      return;
    }

    setAuthStatus(`matched ${match.personnelId} at ${match.similarity.toFixed(3)}`);
    await saveAttendance({
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      personnel_id: match.personnelId,
      embedding,
      timestamp: Date.now(),
      synced: 0,
    });
  };

  useEffect(() => {
    let isMounted = true;

    async function bootstrap() {
      try {
        await initAttendanceDb();
        const faceSession = await ort.InferenceSession.create(FACE_MODEL);
        const livenessSession = await ort.InferenceSession.create(LIVENESS_MODEL);

        if (!isMounted) {
          return;
        }

        faceSessionRef.current = faceSession;
        livenessSessionRef.current = livenessSession;
        setModelStatus('models loaded and attendance DB ready');
      } catch (error) {
        if (isMounted) {
          setModelStatus(`model load failed: ${String(error)}`);
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
        setSyncStatus('back online, syncing offline rows');
        void syncAndPurge(uploadToS3)
          .then(() => setSyncStatus('sync complete'))
          .catch(error => setSyncStatus(`sync failed: ${String(error)}`));
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
        <Text style={styles.statusText}>{modelStatus}</Text>
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
        <Text style={styles.body}>{modelStatus}</Text>
        <Text style={styles.body}>{syncStatus}</Text>
        <Text style={styles.body}>{authStatus}</Text>
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
    const livenessInput = frameToInputTensor(frame, LIVENESS_INPUT_SIZE);
    const livenessResult = await livenessSession.run({ input: livenessInput });
    const livenessLogit = extractScalar(livenessResult);

    if (livenessLogit === null) {
      console.log('liveness score missing from liveness model output');
      return;
    }

    const livenessScore = 1 / (1 + Math.exp(-livenessLogit));
    if (livenessScore < LIVENESS_THRESHOLD) {
      console.log(`liveness rejected: ${livenessScore.toFixed(3)}`);
      return;
    }

    const input = frameToInputTensor(frame, FACE_INPUT_SIZE);
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

function extractScalar(result: ort.InferenceSession.OnnxValueMapType) {
  const output = result.logit ?? result.output ?? result['logit'] ?? result['output'];
  if (!output || !('data' in output)) {
    return null;
  }

  const data = output.data;
  if (data instanceof Float32Array) {
    return data[0] ?? null;
  }

  if (ArrayBuffer.isView(data)) {
    const values = new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
    return values[0] ?? null;
  }

  return null;
}

function frameToInputTensor(frame: any, size: number) {
  const pixelBuffer = frame.getPixelBuffer();
  const pixels = new Uint8Array(pixelBuffer);
  const chw = new Float32Array(3 * size * size);
  const pixelCount = size * size;

  for (let index = 0; index < pixelCount; index += 1) {
    const pixelOffset = index * 4;
    const red = pixels[pixelOffset] / 255;
    const green = pixels[pixelOffset + 1] / 255;
    const blue = pixels[pixelOffset + 2] / 255;

    chw[index] = (red - 0.5) / 0.5;
    chw[pixelCount + index] = (green - 0.5) / 0.5;
    chw[pixelCount * 2 + index] = (blue - 0.5) / 0.5;
  }

  return new ort.Tensor('float32', chw, [1, 3, size, size]);
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
