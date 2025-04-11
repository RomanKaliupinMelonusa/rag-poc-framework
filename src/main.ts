// src/main.ts (or index.ts)
import DatabaseAdapter from './lib/DatabaseAdapter';
import { EmbeddingService } from './services/EmbeddingService';
import { DocumentProcessorService } from './services/DocumentProcessorService';
import path from 'path';
import * as dotenv from 'dotenv';
dotenv.config();

// --- Dependencies Setup ---

// 1. Vercel AI SDK Embedding Model Initialization (Replace with your actual model)
// This requires configuration (API keys etc.) usually via environment variables
import { createOpenAI } from '@ai-sdk/openai'; // Example provider

// IMPORTANT: Handle API keys securely, typically via environment variables
const openai = createOpenAI({
    apiKey: process.env.OPENAI_API_KEY // Example
});
// Choose the specific embedding model you want to use
// Make sure the model identifier is correct for the provider
const embeddingModel = openai.embedding('text-embedding-3-small'); // Example model

// 2. Instantiate Adapters/Services (Dependency Injection)
const dbAdapter = new DatabaseAdapter(); // Uses './dataBase' directory
const embeddingService = new EmbeddingService(embeddingModel);
const docProcessor = new DocumentProcessorService(dbAdapter, embeddingService);

// --- Main Application Logic ---

async function main() {
    try {
        // Example: Process a PDF
        const pdfPath = path.join(__dirname, 'combined.pdf');  // Replace with your actual XML file path

        // You might need to create a dummy PDF or use a real one for testing
        // await fs.writeFile(pdfPath, 'This is the first sentence. This is the second sentence.'); // Create dummy file if needed

        console.log('\n--- Processing PDF ---');
        // const processingResult = await docProcessor.processPdf(pdfPath);
        // console.log(`PDF Processing Result: Resource ID ${processingResult.resource.id}, Embeddings Stored: ${processingResult.embeddingsCount}`);

        // Allow some time for file writes if needed (usually not necessary with await)
        // await new Promise(resolve => setTimeout(resolve, 100));

        // Example: Search for relevant content
        console.log('\n--- Searching Content ---');
        const searchQuery = "Diamond Minimum Color";
        const searchResult = await docProcessor.findRelevantContent(searchQuery);

        if (searchResult) {
            console.log('Search Result Found:\n', searchResult, '\n\n');
            // console.log(`  Similarity: ${(searchResult.similarity * 100).toFixed(2)}%`);
            // console.log(`  Matching Chunk: "${searchResult.sourceContentChunk}"`);
            // console.log(`  Original Doc ID: ${searchResult.originalResourceId}`);
        } else {
            console.log(`No relevant content found for query: "${searchQuery}"`);
        }

    } catch (error) {
        console.error('\n--- MAIN EXECUTION FAILED ---');
        if (error instanceof Error) {
            console.error(error.message);
            // console.error(error.stack); // Uncomment for more detail
        } else {
            console.error("An unknown error occurred in main:", error);
        }
    } finally {
        // Clean up resources if needed (e.g., close DB connections if using a real DB)
        console.log("\nExecution finished.");
    }
}

main();
