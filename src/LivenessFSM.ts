export type ChallengeType = 'BLINK' | 'SMILE' | 'TURN_LEFT' | 'TURN_RIGHT';
export type FsmState = 'IDLE' | 'CHALLENGE_ACTIVE' | 'VERIFIED' | 'FAILED';

export interface FaceDataMetrics {
  leftEar: number;
  rightEar: number;
  smilingProbability: number;
  yaw: number;
}

export interface LivenessChallengeOptions {
  fps?: number;
  blinkAlpha?: number;
  blinkBaselineSeconds?: number;
  blinkMovingAverageSeconds?: number;
  smileThreshold?: number;
  smileConsecutiveFrames?: number;
  turnAngleDegrees?: number;
  challengeDurationSeconds?: number;
}

export class LivenessChallengeStateMachine {
  private currentState: FsmState = 'IDLE';
  private activeChallenge: ChallengeType = 'BLINK';
  private framesSatisfied = 0;
  private challengeTimeout = 0;
  private readonly fps!: number;
  private readonly blinkAlpha!: number;
  private readonly blinkBaselineFrameCount!: number;
  private readonly blinkMovingAverageWindow!: number;
  private readonly smileThreshold!: number;
  private readonly smileConsecutiveFrames!: number;
  private readonly turnAngleDegrees!: number;
  private readonly maxFramesForChallenge!: number;
  private readonly blinkRawHistory: number[] = [];
  private readonly blinkSmoothedHistory: number[] = [];
  private blinkThreshold: number | null = null;
  private blinkBaselineReady = false;

  constructor(
    private onVerified: () => void,
    private onFailed: (reason: string) => void,
    options: LivenessChallengeOptions = {},
  ) {
    this.fps = options.fps ?? 30;
    this.blinkAlpha = options.blinkAlpha ?? 2.0;
    this.blinkBaselineFrameCount = Math.max(1, Math.round((options.blinkBaselineSeconds ?? 1.5) * this.fps));
    this.blinkMovingAverageWindow = Math.max(
      1,
      Math.round((options.blinkMovingAverageSeconds ?? 0.04) * this.fps),
    );
    this.smileThreshold = options.smileThreshold ?? 0.7;
    this.smileConsecutiveFrames = Math.max(1, options.smileConsecutiveFrames ?? 3);
    this.turnAngleDegrees = options.turnAngleDegrees ?? 25.0;
    this.maxFramesForChallenge = Math.max(
      1,
      Math.round((options.challengeDurationSeconds ?? 5) * this.fps),
    );
  }

  public startNewChallenge(challenge: ChallengeType) {
    this.activeChallenge = challenge;
    this.currentState = 'CHALLENGE_ACTIVE';
    this.framesSatisfied = 0;
    this.challengeTimeout = 0;
    this.blinkRawHistory.length = 0;
    this.blinkSmoothedHistory.length = 0;
    this.blinkThreshold = null;
    this.blinkBaselineReady = false;
  }

  public updateMetrics(metrics: FaceDataMetrics) {
    if (this.currentState !== 'CHALLENGE_ACTIVE') {
      return;
    }

    this.challengeTimeout += 1;
    if (this.challengeTimeout > this.maxFramesForChallenge) {
      this.currentState = 'FAILED';
      this.onFailed(`Timeout waiting for challenge: ${this.activeChallenge}`);
      return;
    }

    switch (this.activeChallenge) {
      case 'BLINK':
        this.updateBlinkState(metrics);
        break;

      case 'SMILE':
        if (metrics.smilingProbability >= this.smileThreshold) {
          this.framesSatisfied += 1;
          if (this.framesSatisfied >= this.smileConsecutiveFrames) {
            this.currentState = 'VERIFIED';
            this.onVerified();
          }
        } else {
          this.framesSatisfied = 0;
        }
        break;

      case 'TURN_LEFT':
        if (metrics.yaw > this.turnAngleDegrees) {
          this.currentState = 'VERIFIED';
          this.onVerified();
        }
        break;

      case 'TURN_RIGHT':
        if (metrics.yaw < -this.turnAngleDegrees) {
          this.currentState = 'VERIFIED';
          this.onVerified();
        }
        break;
    }
  }

  public getFsmState(): FsmState {
    return this.currentState;
  }

  public getActiveChallenge(): ChallengeType {
    return this.activeChallenge;
  }

  public getBlinkThreshold(): number | null {
    return this.blinkThreshold;
  }

  private updateBlinkState(metrics: FaceDataMetrics) {
    const bilateralEar = computeBilateralEar(metrics.leftEar, metrics.rightEar);
    const smoothedEar = this.pushAndSmooth(bilateralEar);

    if (!this.blinkBaselineReady) {
      this.blinkSmoothedHistory.push(smoothedEar);
      if (this.blinkSmoothedHistory.length >= this.blinkBaselineFrameCount) {
        this.blinkThreshold = computeBlinkThreshold(
          this.blinkSmoothedHistory,
          this.blinkAlpha,
        );
        this.blinkBaselineReady = true;
        this.framesSatisfied = 0;
      }
      return;
    }

    if (this.blinkThreshold === null) {
      this.blinkThreshold = computeBlinkThreshold(this.blinkSmoothedHistory, this.blinkAlpha);
    }

    const minimumBlinkFrames = Math.max(2, Math.round(0.033 * this.fps));
    if (smoothedEar < this.blinkThreshold) {
      this.framesSatisfied += 1;
      if (this.framesSatisfied >= minimumBlinkFrames) {
        this.currentState = 'VERIFIED';
        this.onVerified();
      }
      return;
    }

    this.framesSatisfied = 0;
  }

  private pushAndSmooth(rawEar: number) {
    this.blinkRawHistory.push(rawEar);
    if (this.blinkRawHistory.length > this.blinkMovingAverageWindow) {
      this.blinkRawHistory.shift();
    }

    const total = this.blinkRawHistory.reduce((sum, value) => sum + value, 0);
    return total / this.blinkRawHistory.length;
  }
}

function computeBilateralEar(leftEar: number, rightEar: number) {
  return (leftEar + rightEar) / 2;
}

function computeBlinkThreshold(values: number[], alpha: number) {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const median =
    sorted.length % 2 === 1
      ? sorted[(sorted.length - 1) / 2]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;

  if (values.length < 2) {
    return median;
  }

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) * (value - mean), 0) /
    (values.length - 1);

  return median - alpha * Math.sqrt(variance);
}