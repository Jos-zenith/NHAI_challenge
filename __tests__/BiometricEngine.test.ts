import { computeCosineSimilarity } from '../src/embedding_matcher';
import {
  FaceDataMetrics,
  LivenessChallengeStateMachine,
} from '../src/LivenessFSM';

describe('DatalakeFace Vector Matching Unit Tests', () => {
  it('should compute exact cosine similarity for identical vectors', () => {
    const mockVector = new Float32Array([1.0, 0.0, 0.0, 0.0]);
    const score = computeCosineSimilarity(mockVector, mockVector);
    expect(score).toBeCloseTo(1.0, 5);
  });

  it('should compute zero similarity for orthogonal vectors', () => {
    const vectorA = new Float32Array([1.0, 0.0, 0.0]);
    const vectorB = new Float32Array([0.0, 1.0, 0.0]);
    const score = computeCosineSimilarity(vectorA, vectorB);
    expect(score).toBe(0.0);
  });

  it('should throw an error if vectors have mismatched dimensions', () => {
    const vectorA = new Float32Array([1.0, 2.0]);
    const vectorB = new Float32Array([1.0, 2.0, 3.0]);
    expect(() => computeCosineSimilarity(vectorA, vectorB)).toThrow();
  });
});

describe('Liveness FSM Transition Tests', () => {
  const baselineMetrics: FaceDataMetrics = {
    leftEar: 0.35,
    rightEar: 0.35,
    smilingProbability: 0.1,
    yaw: 2.0,
  };

  it('should successfully transition to VERIFIED after satisfying consecutive blink frames', () => {
    const onVerifiedMock = jest.fn();
    const onFailedMock = jest.fn();

    const fsm = new LivenessChallengeStateMachine(onVerifiedMock, onFailedMock, {
      fps: 2,
      blinkBaselineSeconds: 0.5,
      blinkMovingAverageSeconds: 0.5,
    });
    fsm.startNewChallenge('BLINK');

    const lowEarMetrics: FaceDataMetrics = {
      leftEar: 0.12,
      rightEar: 0.12,
      smilingProbability: 0.1,
      yaw: 2.0,
    };

    fsm.updateMetrics(baselineMetrics);
    expect(fsm.getFsmState()).toBe('CHALLENGE_ACTIVE');

    fsm.updateMetrics(lowEarMetrics);
    fsm.updateMetrics(lowEarMetrics);

    expect(fsm.getFsmState()).toBe('VERIFIED');
    expect(onVerifiedMock).toHaveBeenCalledTimes(1);
    expect(onFailedMock).not.toHaveBeenCalled();
  });

  it('should fail if the timeout frame limit is exceeded', () => {
    const onVerifiedMock = jest.fn();
    const onFailedMock = jest.fn();

    const fsm = new LivenessChallengeStateMachine(onVerifiedMock, onFailedMock, {
      fps: 10,
      challengeDurationSeconds: 0.1,
    });
    fsm.startNewChallenge('BLINK');

    for (let index = 0; index < 2; index += 1) {
      fsm.updateMetrics(baselineMetrics);
    }

    expect(fsm.getFsmState()).toBe('FAILED');
    expect(onFailedMock).toHaveBeenCalledWith(expect.stringContaining('Timeout'));
  });
});