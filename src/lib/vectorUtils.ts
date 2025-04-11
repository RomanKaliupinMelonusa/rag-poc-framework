/**
 * Calculates the dot product of two vectors.
 * @param {number[]} vecA - The first vector.
 * @param {number[]} vecB - The second vector.
 * @returns {number} The dot product.
 * @throws {Error} If vectors have different dimensions.
 */
function dotProduct(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
      throw new Error("Vectors must have the same dimension for dot product.");
    }
    let product = 0;
    for (let i = 0; i < vecA.length; i++) {
      product += vecA[i] * vecB[i];
    }
    return product;
  }

  /**
   * Calculates the magnitude (Euclidean norm) of a vector.
   * @param {number[]} vec - The vector.
   * @returns {number} The magnitude of the vector.
   */
  function magnitude(vec: number[]): number {
    let sumOfSquares = 0;
    for (let i = 0; i < vec.length; i++) {
      sumOfSquares += vec[i] * vec[i];
    }
    return Math.sqrt(sumOfSquares);
  }

  /**
   * Calculates the cosine similarity between two vectors.
   * Result ranges from -1 (opposite) to 1 (identical), 0 (orthogonal).
   * @param {number[]} vecA - The first vector.
   * @param {number[]} vecB - The second vector.
   * @returns {number} The cosine similarity. Returns 0 if either vector has zero magnitude or dimensions mismatch.
   */
  export function cosineSimilarity(vecA: number[], vecB: number[]): number {
    try {
      const magA = magnitude(vecA);
      const magB = magnitude(vecB);

      if (magA === 0 || magB === 0 || vecA.length !== vecB.length) {
        return 0; // Or handle dimension mismatch more strictly if needed
      }

      const dot = dotProduct(vecA, vecB);
      return dot / (magA * magB);
    } catch (error) {
      console.error("Error calculating cosine similarity:", error);
      return 0; // Return neutral value on error
    }
  }
