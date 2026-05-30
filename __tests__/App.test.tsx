/**
 * @format
 */

import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import App from '../App';

jest.mock('onnxruntime-react-native');
jest.mock('@react-native-community/netinfo', () => ({
  __esModule: true,
  default: {
    addEventListener: jest.fn(() => jest.fn()),
    fetch: jest.fn(async () => ({ isConnected: false })),
  },
}));
jest.mock('react-native-worklets', () => ({
  scheduleOnRN: jest.fn((callback: (...args: unknown[]) => unknown, ...args: unknown[]) =>
    callback(...args)),
}));
jest.mock('react-native-vision-camera', () => {
  const React = require('react');

  return {
    Camera: (props: React.PropsWithChildren<Record<string, unknown>>) =>
      React.createElement('Camera', props, props.children),
    useCameraDevice: jest.fn(() => ({ id: 'front-camera' })),
    useCameraPermission: jest.fn(() => ({
      hasPermission: true,
      requestPermission: jest.fn(async () => true),
    })),
    useFrameOutput: jest.fn(() => ({})),
  };
});

test('renders correctly', async () => {
  await ReactTestRenderer.act(() => {
    ReactTestRenderer.create(<App />);
  });
});
