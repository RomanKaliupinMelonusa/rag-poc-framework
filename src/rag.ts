// src/vector-search-updated.ts

import fs from "node:fs/promises";
import path from "node:path";
import pdfParse from "pdf-parse";
import {
    // Core components are usually top-level
    VectorStoreIndex,
    storageContextFromDefaults,
    Document,
    Metadata,
    NodeWithScore,
    Settings,
    MetadataMode,
    OpenAIEmbedding,
    BaseQueryEngine,
    ChatMessage,
    StorageContext,
    OpenAI
} from "llamaindex";


// Optional: Load environment variables (e.g., OPENAI_API_KEY for default models)
import dotenv from "dotenv";
dotenv.config();

// Optional: Configure settings globally (e.g., if not using OpenAI defaults)
// import { Ollama } from "llamaindex/llms/ollama";
// Settings.llm = new Ollama({ model: "llama2" });
Settings.embedModel = Settings.embedModel = new OpenAIEmbedding({
    model: "text-embedding-3-small", // Your chosen embedding model
    apiKey: process.env.OPENAI_API_KEY
}); // Configure embedding model if needed


if (process.env.PERFORM_RAG === 'true' ) {
    Settings.llm = new OpenAI({
        model: process.env.LLM_MODEL,
        apiKey: process.env.OPENAI_API_KEY
    });
    console.log(`LLM configured for RAG: ${Settings.llm.metadata.model}`);
}

// --- Configuration ---
const DEFAULT_INDEX_DIR = "./storage";
const DEFAULT_DOCUMENTS_DIR = "./documents";

// --- Interfaces ---
interface SearchResult {
    text: string;
    score: number;
    metadata?: Metadata;
}

// --- Helper Functions --- (ensureDirectoryExists remains the same)
async function ensureDirectoryExists(dirPath: string): Promise<void> {
    try {
        await fs.access(dirPath);
    } catch (error: any) {
        if (error.code === 'ENOENT') {
            await fs.mkdir(dirPath, { recursive: true });
            console.log(`Created directory: ${dirPath}`);
        } else {
            console.error(`Error accessing directory ${dirPath}:`, error);
            throw error;
        }
    }
}

// --- Core Logic ---

/**
 * Loads and parses PDF documents manually from a specified directory.
 * SRP: Responsible for finding PDF files, parsing them, and creating Document objects.
 * @param documentsPath Path to the directory containing PDFs.
 * @returns Array of LlamaIndex Document objects.
 * @throws Error if the directory cannot be read.
 */
async function loadDocuments(documentsPath: string): Promise<Document[]> {
    console.log(`Manually loading documents from: ${documentsPath}`);
    await ensureDirectoryExists(documentsPath); // Ensure target dir exists

    const loadedDocuments: Document[] = [];
    try {
        const fileNames = await fs.readdir(documentsPath);
        console.log(`Found ${fileNames.length} files/folders in directory.`);

        for (const fileName of fileNames) {
            const filePath = path.join(documentsPath, fileName);
            const fileStats = await fs.stat(filePath);

            // Check if it's a file and has a .pdf extension
            if (fileStats.isFile() && path.extname(fileName).toLowerCase() === '.pdf') {
                console.log(`Processing PDF: ${fileName}`);
                try {
                    // Read the raw PDF file content as a buffer
                    const dataBuffer = await fs.readFile(filePath);

                    // Parse the PDF buffer using pdf-parse
                    const pdfData = await pdfParse(dataBuffer);

                    // Create a LlamaIndex Document
                    // We use filePath as id_ for easy tracking
                    // pdfData.text contains the extracted text content
                    const document = new Document({
                        text: pdfData.text || "", // Use extracted text, default to empty string if none
                        id_: filePath,             // Use file path as the document ID
                        metadata: {                // Add relevant metadata
                            file_path: filePath,
                            file_name: fileName,
                            // pdfData also contains 'numpages', 'numrender', 'info', 'metadata', 'version'
                            // You can add more metadata fields if needed, e.g.:
                            // num_pages: pdfData.numpages
                        }
                    });

                    loadedDocuments.push(document);
                    console.log(`Successfully created Document for: ${fileName}`);

                } catch (parseError: any) {
                    console.error(`Failed to parse PDF file ${fileName}:`, parseError.message || parseError);
                    // Decide whether to skip the file or throw an error
                    // Continue for now, just log the error
                }
            } else if (!fileStats.isDirectory()) {
                console.log(`Skipping non-PDF file: ${fileName}`);
            }
            // Optionally handle directories if needed
        }

        if (loadedDocuments.length === 0) {
            console.warn(`No PDF documents were successfully loaded from ${documentsPath}.`);
        } else {
            console.log(`Successfully loaded and parsed ${loadedDocuments.length} PDF document(s).`);
        }
        return loadedDocuments;

    } catch (error) {
        console.error(`Failed to read directory ${documentsPath}:`, error);
        throw new Error(`Failed to read documents directory. Check permissions and path.`);
    }
}

// createAndPersistIndex - Removed serviceContextFromDefaults
async function createAndPersistIndex(documents: Document[], persistDir: string): Promise<VectorStoreIndex> {
    console.log(`Creating index and persisting to: ${persistDir}`);
    await ensureDirectoryExists(persistDir);

    try {
        // LlamaIndex now uses global Settings or defaults if not specified here
        const storageContext = await storageContextFromDefaults({ persistDir });

        // No need to pass serviceContext explicitly if relying on defaults/global Settings
        const index = await VectorStoreIndex.fromDocuments(documents, {
            storageContext,
        });
        console.log(`Index created and persisted successfully.`);
        return index;
    } catch (error) {
        console.error(`Failed to create or persist index at ${persistDir}:`, error);
        throw new Error(`Index creation/persistence failed. Check console for details.`);
    }
}

// loadIndex - Removed serviceContextFromDefaults
async function loadIndex(persistDir: string): Promise<VectorStoreIndex> {
    console.log(`Attempting to load index from: ${persistDir}`);
    try {
        await fs.access(persistDir);
        const storageContext = await storageContextFromDefaults({ persistDir });

        // No need to pass serviceContext explicitly if relying on defaults/global Settings
        const index = await VectorStoreIndex.init({
            storageContext,
        });
        console.log("Index loaded successfully.");
        return index;
    } catch (error) {
        console.error(`Failed to load index from ${persistDir}. It might not exist or be corrupted.`, error);
        throw new Error(`Failed to load index. Ensure it was created previously at ${persistDir}.`);
    }
}

// getOrCreateIndex remains the same conceptually
async function getOrCreateIndex(indexDir: string = DEFAULT_INDEX_DIR, documentsDir: string = DEFAULT_DOCUMENTS_DIR): Promise<VectorStoreIndex> {
    try {
        const index = await loadIndex(indexDir);
        return index;
    } catch (error: any) {
        if (error.message.includes("Failed to load index") || error.code === 'ENOENT') {
            console.log("Index not found. Creating a new one...");
            const documents = await loadDocuments(documentsDir);
            if (documents.length === 0) {
                throw new Error("Cannot create index: No documents were loaded.");
            }
            const index = await createAndPersistIndex(documents, indexDir);
            return index;
        } else {
            console.error("An unexpected error occurred while trying to load the index:", error);
            throw error;
        }
    }
}


// performSearch - Removed explicit QueryEngine type annotation
async function performSearch(index: VectorStoreIndex, queryText: string, topN: number): Promise<SearchResult[]> {
    console.log(`Performing search for "${queryText}" (top ${topN})...`);
    try {
        // Let TypeScript infer the type of queryEngine
        const queryEngine = index.asQueryEngine({
            similarityTopK: topN,
        });

        const results = await queryEngine.query({ query: queryText });

        const sourceNodes: NodeWithScore[] = results.sourceNodes || [];

        const searchResults: SearchResult[] = sourceNodes.map(nodeWithScore => ({
            text: nodeWithScore.node.getContent(MetadataMode.LLM),
            score: nodeWithScore.score ?? 0,
            metadata: nodeWithScore.node.metadata,
        }));

        console.log(`Search completed. Found ${searchResults.length} results.`);
        return searchResults;

    } catch (error) {
        console.error("Error during search:", error);
        throw new Error("Search operation failed.");
    }
}

/**
 * Loads an index from a persistence directory and performs a query.
 *
 * @param queryText The user's query string.
 * @param persistDir The directory path where the index is saved.
 * @returns Promise<{ response: string; sourceNodes: NodeWithScore[] | undefined }> The query response and source nodes.
 * @throws Error if the persistence directory does not exist.
 */
async function queryIndex(
    index: VectorStoreIndex,
    queryText: string
  ): Promise<{ message: ChatMessage; sourceNodes: NodeWithScore[] | undefined }> {
    console.log(`\n--- Starting Query Process ---`);
    console.log(`Query: "${queryText}"`);

    try {
      // Create a query engine from the loaded index
      const queryEngine: BaseQueryEngine = index.asQueryEngine();

      // Perform the query
      console.log("Performing query...");
      const { message, sourceNodes } = await queryEngine.query({
        query: queryText,
      });
      console.log("Query completed.");
      console.log(`--- Query Process Finished ---`);


      // Return the relevant parts of the result
      return {
          message,
          sourceNodes
      };

    } catch (error) {
      console.error("Error during query:", error);
      throw error; // Re-throw error after logging
    }
  }

// --- Example Usage --- (main function remains the same)
async function main() {
    const indexDirectory = path.resolve(process.env.INDEX_DIR || DEFAULT_INDEX_DIR);
    const documentsDirectory = path.resolve(process.env.DOCUMENTS_DIR || DEFAULT_DOCUMENTS_DIR);
    const searchQuery = process.env.QUERY || "What are the main challenges mentioned?";
    const numberOfResults = parseInt(process.env.TOP_N || "3", 10);

    console.log("--- Vector Search Tool (Updated) ---");
    console.log(`Index directory: ${indexDirectory}`);
    console.log(`Documents directory: ${documentsDirectory}`);
    console.log(`Search query: "${searchQuery}"`);
    console.log(`Number of results: ${numberOfResults}`);
    console.log("--------------------------");

    try {
        const index = await getOrCreateIndex(indexDirectory, documentsDirectory);

        if (process.env.PERFORM_RAG === 'true') {
            const { message, sourceNodes } = await queryIndex(index, searchQuery);

            console.log("\nQuery Response:");
            console.log("Output response with sources: \n", message.content, '\n\n');
        } else {
            const searchResults = await performSearch(index, searchQuery, numberOfResults);

            console.log("\n--- Search Results ---");
            if (searchResults.length > 0) {
                searchResults.forEach((result, i) => {
                    console.log(`\n[${i + 1}] Score: ${result.score.toFixed(4)}`);
                    // console.log(`Metadata: ${JSON.stringify(result.metadata)}`);
                    console.log("Text:");
                    console.log(result.text.substring(0, 500) + (result.text.length > 500 ? "..." : ""));
                    console.log("---");
                });
            } else {
                console.log("No relevant results found for your query.");
            }
        }
    } catch (error) {
        console.error("\n--- An error occurred during execution ---");
        console.error(error);
        process.exitCode = 1;
    }
}

main();
