import { useCallback, useState } from 'react';
import { ChallengeType } from './LivenessFSM';

export type BiometricProgressState =
  | 'INITIALIZING'
  | 'READY'
  | 'LIVENESS_RUNNING'
  | 'EMBEDDING_RUNNING'
  | 'AUTH_SUCCESS'
  | 'AUTH_FAILED'
  | 'SYNCING';

export interface UIStatusState {
  progressState: BiometricProgressState;
  activeChallenge: ChallengeType | null;
  errorMessage: string | null;
  similarityScore: number | null;
  unsyncedRowsCount: number;
}

export function useBiometricUIState() {
  const [uiState, setUiState] = useState<UIStatusState>({
    progressState: 'INITIALIZING',
    activeChallenge: null,
    errorMessage: null,
    similarityScore: null,
    unsyncedRowsCount: 0,
  });

  const transitionTo = useCallback(
    (nextState: BiometricProgressState, patch: Partial<UIStatusState> = {}) => {
      setUiState(current => ({
        ...current,
        progressState: nextState,
        activeChallenge:
          patch.activeChallenge !== undefined ? patch.activeChallenge : current.activeChallenge,
        errorMessage:
          patch.errorMessage !== undefined ? patch.errorMessage : current.errorMessage,
        similarityScore:
          patch.similarityScore !== undefined ? patch.similarityScore : current.similarityScore,
        unsyncedRowsCount:
          patch.unsyncedRowsCount !== undefined ? patch.unsyncedRowsCount : current.unsyncedRowsCount,
      }));
    },
    [],
  );

  return {
    uiState,
    transitionTo,
  };
}