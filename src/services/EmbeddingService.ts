// src/services/EmbeddingService.ts
import { embed, EmbeddingModel } from 'ai'; // Vercel AI SDK - embedding
import { cosineSimilarity } from '../lib/vectorUtils'; // Assume a helper utility exists
import { BaseRecord } from '../interfaces/baseRecord.types';


// Define the structure expected for embedding records (used for search comparison)
// This might live in a shared types file
export interface EmbeddingRecord extends BaseRecord {
    // id: string; // from BaseRecord
    resourceId: string; // Link to the original resource
    contentChunk: string; // The text chunk that was embedded
    embedding: number[]; // The actual vector embedding
}

export class EmbeddingService {
    // Dependency: Inject the specific embedding model client
    private embeddingModel: EmbeddingModel<any>;

    constructor(embeddingModel: EmbeddingModel<any>) {
        this.embeddingModel = embeddingModel;
    }

    /**
     * Generates a single vector embedding for the given text.
     * @param {string} value - The text to embed.
     * @returns {Promise<number[]>} The generated vector embedding.
     */
    async generateEmbedding(value: string): Promise<number[]> {
        if (!value) {
            return []; // Or throw error
        }
        try {
            const { embedding } = await embed({
                model: this.embeddingModel,
                value: value,
            });
            return embedding;
        } catch (error) {
            console.error("Error generating single embedding:", error);
            throw new Error(`Failed to generate embedding: ${(error as Error).message}`);
        }
    }

    /**
     * Chunks text and generates embeddings for each chunk.
     * @param {string} value - The full text content.
     * @param {number} [chunkSize=500] - Approximate size for text chunks.
     * @param {number} [overlap=50] - Overlap between chunks.
     * @returns {Promise<Array<{ content: string; embedding: number[] }>>} An array of objects containing text chunks and their embeddings.
     */
    async generateEmbeddings(value: string, chunkSize: number = 500, overlap: number = 50): Promise<Array<{ content: string; embedding: number[] }>> {
        if (!value) {
            return [];
        }
        // --- Basic Chunking Logic (Replace with a more robust library if needed) ---
        const chunks: string[] = [];
        for (let i = 0; i < value.length; i += chunkSize - overlap) {
            chunks.push(value.substring(i, i + chunkSize));
        }
        // --------------------------------------------------------------------------

        if (chunks.length === 0) {
            return [];
        }

        try {
            // Use embedMany for potential efficiency (check if Vercel SDK supports this well)
            // Or loop and call embed individually if embedMany isn't suitable/available
            // For simplicity here, let's loop:
            const results: Array<{ content: string; embedding: number[] }> = [];
            for (const chunk of chunks) {
                const { embedding } = await embed({
                    model: this.embeddingModel,
                    value: chunk
                });
                results.push({ content: chunk, embedding: embedding });
            }
            return results;

            /* // Example using embedMany if available and desired:
            const { embeddings } = await embedMany({
                model: this.embeddingModel,
                values: chunks,
            });
            return chunks.map((chunk, i) => ({ content: chunk, embedding: embeddings[i] }));
            */
        } catch (error) {
            console.error("Error generating multiple embeddings:", error);
            throw new Error(`Failed to generate embeddings: ${(error as Error).message}`);
        }
    }

    /**
     * Finds the stored embedding record most similar to the user query's embedding.
     * Note: This method only performs the comparison logic; it expects the query vector
     * and the stored records to be provided.
     * @param {number[]} queryVector - The embedding vector of the user's query.
     * @param {EmbeddingRecord[]} storedEmbeddings - An array of stored embedding records (from DB).
     * @returns {EmbeddingRecord | null} The stored embedding record with the highest cosine similarity, or null if none found.
     */
    findMostSimilar(queryVector: number[], storedEmbeddings: EmbeddingRecord[]): EmbeddingRecord[] | null {
        if (!queryVector || queryVector.length === 0 || !storedEmbeddings || storedEmbeddings.length === 0) {
            return null;
        }

        const SIMILARITY_THRESHOLD = 0.5;
        const similarityResults: EmbeddingRecord[] = [];
        let bestMatch: EmbeddingRecord | null = null;
        let highestSimilarity = -Infinity; // Cosine similarity ranges from -1 to 1

        for (const record of storedEmbeddings) {
            if (!record.embedding || record.embedding.length !== queryVector.length) {
                 console.warn(`Skipping record ID ${record.id} due to missing or mismatched embedding dimension.`);
                 continue; // Skip if embedding is invalid or dimensions don't match
            }

            const similarity = cosineSimilarity(queryVector, record.embedding);

            // if (similarity > highestSimilarity) {
            //     highestSimilarity = similarity;
            //     bestMatch = record;
            // }
            if (similarity >= SIMILARITY_THRESHOLD) {
                similarityResults.push(record);
            }
        }

        // Optional: Add a threshold check
        // const SIMILARITY_THRESHOLD = 0.7;
        // if (highestSimilarity < SIMILARITY_THRESHOLD) {
        //     return null;
        // }

        // console.log(`Highest similarity found: ${highestSimilarity.toFixed(4)} for record ID ${bestMatch?.id}`);
        // return bestMatch;

        return similarityResults;
    }
}
