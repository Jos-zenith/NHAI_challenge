export class Tensor {
  constructor(public type: string, public data: Float32Array, public dims: number[]) {}
}

export const InferenceSession = {
  create: jest.fn().mockImplementation(async () => {
    return {
      inputNames: ['input_node'],
      outputNames: ['output_node'],
      run: jest.fn().mockImplementation(async () => {
        const mockData = new Float32Array([0.05, 0.95]);
        return {
          output_node: {
            data: mockData,
          },
        };
      }),
    };
  }),
};