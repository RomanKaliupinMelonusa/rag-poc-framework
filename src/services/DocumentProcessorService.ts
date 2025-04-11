// src/services/DocumentProcessorService.ts
import { DatabaseAdapter } from '@/lib/DatabaseAdapter'; // Assuming BaseRecord is exported or defined here
import { EmbeddingService, EmbeddingRecord } from './EmbeddingService';
import { BaseRecord } from '../interfaces/baseRecord.types';
import { cosineSimilarity } from '@/lib/vectorUtils';

// Assume pdf-parse or similar library for PDF reading
import pdfParse from 'pdf-parse'; // Example dependency
import fs from 'fs/promises';
import path from 'path';

// Define structure for search results returned to the user
// export interface SearchServiceResult {
//     sourceContentChunk: string; // The best matching text chunk
//     originalResourceId: string; // ID of the full document
//     similarity: number; // The similarity score
// }

export interface SearchServiceResult {
    chunks: any[]; // The best matching text chunks
}

export class DocumentProcessorService {
    // Dependencies injected via constructor (DIP)
    private dbAdapter: DatabaseAdapter; // Using concrete class for simplicity, could be interface IDatabaseAdapter
    private embeddingService: EmbeddingService; // Could be interface IEmbeddingService

    // Constants for table names
    private readonly RESOURCES_TABLE = 'resources';
    private readonly EMBEDDINGS_TABLE = 'embeddings';

    constructor(dbAdapter: DatabaseAdapter, embeddingService: EmbeddingService) {
        this.dbAdapter = dbAdapter;
        this.embeddingService = embeddingService;
    }

    /**
     * Processes a PDF file: reads text, stores it, generates embeddings for chunks,
     * and stores embeddings linked to the original resource.
     * @param {string} pdfFilePath - Path to the PDF file.
     * @returns {Promise<{resource: BaseRecord, embeddingsCount: number}>} Info about the processed resource.
     */
    async processPdf(pdfFilePath: string): Promise<{ resource: BaseRecord, embeddingsCount: number }> {
        console.log(`Processing PDF: ${pdfFilePath}`);
        // 1. Read PDF Content
        let fileBuffer: Buffer;
        try {
            fileBuffer = await fs.readFile(pdfFilePath);
        } catch (error) {
             console.error(`Error reading PDF file at ${pdfFilePath}:`, error);
            throw new Error(`Could not read PDF file: ${(error as Error).message}`);
        }

        let pdfData;
        try {
            pdfData = await pdfParse(fileBuffer);
        } catch (error) {
            console.error(`Error parsing PDF file ${pdfFilePath}:`, error);
            throw new Error(`Could not parse PDF content: ${(error as Error).message}`);
        }
        const fullTextContent = pdfData.text;

        if (!fullTextContent || fullTextContent.trim().length === 0) {
             console.warn(`PDF ${pdfFilePath} contains no text content.`);
             // Decide how to handle empty PDFs, e.g., throw error or return specific status
             throw new Error("PDF contains no text content.");
        }

        console.log(`Extracted ${fullTextContent.length} characters of text.`);

        // 2. Store the Full Resource Content
        const resource = await this.dbAdapter.insert(this.RESOURCES_TABLE, {
            content: fullTextContent,
            sourceFile: path.basename(pdfFilePath) // Store original filename maybe
        });
        console.log(`Stored resource with ID: ${resource.id}`);

        // 3. Generate Embeddings for Chunks
        const chunkEmbeddings = await this.embeddingService.generateEmbeddings(fullTextContent);
        console.log(`Generated ${chunkEmbeddings.length} embeddings for the resource.`);

        if (chunkEmbeddings.length === 0) {
            console.warn(`No embeddings generated for resource ${resource.id}.`);
             // Potentially delete the resource record if no embeddings could be made?
             // Or just return successfully but indicate 0 embeddings stored.
             return { resource, embeddingsCount: 0 };
        }

        // 4. Prepare Embedding Records for Storage
        const embeddingRecordsToInsert = chunkEmbeddings.map(ce => ({
            resourceId: resource.id, // Link back to the parent resource
            contentChunk: ce.content,
            embedding: ce.embedding,
        }));

        // 5. Store Embedding Records (using insertMany)
        const insertedEmbeddings = await this.dbAdapter.insertMany(this.EMBEDDINGS_TABLE, embeddingRecordsToInsert);
        console.log(`Successfully stored ${insertedEmbeddings.length} embedding records.`);

        return { resource, embeddingsCount: insertedEmbeddings.length };
    }

    /**
     * Finds content chunks relevant to a user query using vector similarity search.
     * @param {string} userQuery - The user's search query.
     * @returns {Promise<SearchServiceResult | null>} The most relevant result or null if no suitable match found.
     */
    async findRelevantContent(userQuery: string): Promise<SearchServiceResult | null> {
        console.log(`Finding relevant content for query: "${userQuery}"`);

        // 1. Generate Embedding for the User Query
        const queryVector = await this.embeddingService.generateEmbedding(userQuery);
        if (!queryVector || queryVector.length === 0) {
            console.warn("Could not generate embedding for the query.");
            return null;
        }
        console.log(`Generated query vector (dimensions: ${queryVector.length})`);

        // 2. Retrieve All Stored Embeddings from the Database
        // Note: This is inefficient for large datasets! Real vector DBs are optimized for this.
        // Our file DB requires loading everything into memory.
        const allStoredEmbeddings = await this.dbAdapter.getAll(this.EMBEDDINGS_TABLE);
        // We need to cast or validate that these are indeed EmbeddingRecords
        const validStoredEmbeddings = allStoredEmbeddings.filter((r): r is EmbeddingRecord =>
            r && typeof r === 'object' && typeof r.resourceId === 'string' &&
            typeof r.contentChunk === 'string' && Array.isArray(r.embedding)
        );

        if (validStoredEmbeddings.length === 0) {
            console.log("No stored embeddings found in the database to search against.");
            return null;
        }
        console.log(`Comparing query against ${validStoredEmbeddings.length} stored embeddings.`);


        // 3. Find the Most Similar Embedding Record
        // const bestMatches = this.embeddingService.findMostSimilar(queryVector, validStoredEmbeddings);

        // if (!bestMatch) {
        //     console.log("No sufficiently similar content found.");
        //     return null;
        // }

        const matchedChunks = this.embeddingService.findMostSimilar(queryVector, validStoredEmbeddings);

        if (!matchedChunks || matchedChunks.length === 0) {
            console.log("No sufficiently similar content found.");
            return null;
        }

        // 4. Calculate Similarity Score (findMostSimilar doesn't return it directly)
        // We need to recalculate it here, or modify findMostSimilar to return it.
        // const similarityScore = cosineSimilarity(queryVector, bestMatch.embedding);

        // console.log(`Best match found: Embedding ID ${bestMatch.id}, Resource ID ${bestMatch.resourceId}, Similarity ${similarityScore.toFixed(4)}`);

        // 5. Format and Return the Result
        // const result: SearchServiceResult = {
        //     sourceContentChunk: bestMatch.contentChunk,
        //     originalResourceId: bestMatch.resourceId,
        //     similarity: similarityScore,
        // };
        const result: SearchServiceResult = {
            chunks: matchedChunks
        };

        // Optional: Fetch the full original resource content if needed
        // const originalResource = await this.dbAdapter.getById(this.RESOURCES_TABLE, bestMatch.resourceId);
        // if (originalResource) { result.originalFullContent = originalResource.content; }

        return result;
    }
}
