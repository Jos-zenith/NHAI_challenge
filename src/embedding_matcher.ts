export function computeCosineSimilarity(vectorA: Float32Array, vectorB: Float32Array): number {
  if (vectorA.length !== vectorB.length) {
    throw new Error(
      `Dimension mismatch: Vector A (${vectorA.length}) versus Vector B (${vectorB.length})`,
    );
  }

  let dotProduct = 0.0;
  for (let index = 0; index < vectorA.length; index += 1) {
    dotProduct += vectorA[index] * vectorB[index];
  }

  return dotProduct;
}