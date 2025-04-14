import fs from "node:fs/promises";
import path from "node:path"; // Using path module for robust path handling
import * as dotenv from 'dotenv';
dotenv.config();

import {
  Document,
  MetadataMode,
  NodeWithScore,
  VectorStoreIndex,
  storageContextFromDefaults,
  StorageContext, // Types for clarity
  BaseQueryEngine, // Type for query engine
  ChatMessage,
  Settings,
  OpenAI,     // Import OpenAI LLM
  OpenAIEmbedding,
  BaseRetriever
} from "llamaindex";

// --- Function to Index Document and Persist Locally ---

// Do this *before* calling indexDocument or queryIndex

// Get key from environment (safer)
const apiKey = process.env.OPENAI_API_KEY;

if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set in environment variables");
}

// Optionally, configure the LLM as well if needed for querying later
// 1. Configure the Embedding Model
Settings.embedModel = new OpenAIEmbedding({
    model: "text-embedding-3-large", // Your chosen embedding model
    apiKey: apiKey
});

// 2. Configure the Language Model (LLM) for Chat/Response Generation
Settings.llm = new OpenAI({
    apiKey: apiKey
});

/**
 * Indexes a document from a file path and saves the index to a persistence directory.
 * If the persistence directory already exists, it skips indexing.
 *
 * @param filePath The path to the document file to index.
 * @param persistDir The directory path to save the index to.
 * @returns Promise<void>
 */
async function indexDocument(filePath: string, persistDir: string): Promise<void> {
  console.log(`\n--- Starting Indexing Process ---`);
  console.log(`Document path: ${filePath}`);
  console.log(`Persistence directory: ${persistDir}`);

  try {
    // Check if the persistence directory already exists
    await fs.stat(persistDir);
    console.log(`Persistence directory '${persistDir}' already exists. Skipping indexing.`);
    console.log(`--- Indexing Process Finished (Skipped) ---`);
    return; // Exit if index already exists
  } catch (error: any) {
    // If stat fails, it likely means the directory doesn't exist (ENOENT)
    if (error.code !== 'ENOENT') {
      console.error("Error checking persistence directory:", error);
      throw error; // Re-throw unexpected errors
    }
    // Directory doesn't exist, proceed with indexing
    console.log(`Persistence directory '${persistDir}' not found. Proceeding with indexing...`);
  }

  try {
      // Ensure the document file exists before reading
      await fs.stat(filePath);

      // Load the document content from the file
      const fileContent = await fs.readFile(filePath, "utf-8");

      // Create a LlamaIndex Document object
      const document = new Document({ text: fileContent, id_: filePath });

      // Create a StorageContext configured to persist to the specified directory
      // LlamaIndex will automatically create the directory if it doesn't exist.
      const storageContext: StorageContext = await storageContextFromDefaults({
        persistDir: persistDir,
      });

      // Create and build the VectorStoreIndex from the document.
      // The index will be automatically saved to 'persistDir' due to the storageContext.
      console.log("Creating index from document...");
      await VectorStoreIndex.fromDocuments([document], { storageContext });

      console.log(`Index created successfully and saved to '${persistDir}'.`);
      console.log(`--- Indexing Process Finished ---`);

  } catch(error: any) {
      if (error.code === 'ENOENT') {
          console.error(`Error: Document file not found at '${filePath}'`);
      } else {
          console.error("Error during indexing:", error);
      }
      // Optional: re-throw if you want the main flow to stop
      // throw error;
      console.log(`--- Indexing Process Failed ---`);
  }
}

// --- Function to Load Index and Perform Query ---

/**
 * Loads an index from a persistence directory and performs a query.
 *
 * @param queryText The user's query string.
 * @param persistDir The directory path where the index is saved.
 * @returns Promise<{ response: string; sourceNodes: NodeWithScore[] | undefined }> The query response and source nodes.
 * @throws Error if the persistence directory does not exist.
 */
async function queryIndex(
  queryText: string,
  persistDir: string
): Promise<{ message: ChatMessage; sourceNodes: NodeWithScore[] | undefined }> {
  console.log(`\n--- Starting Query Process ---`);
  console.log(`Persistence directory: ${persistDir}`);
  console.log(`Query: "${queryText}"`);

  try {
    // Check if the persistence directory exists. Querying requires it.
    await fs.stat(persistDir);
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      console.error(`Error: Persistence directory '${persistDir}' not found. Cannot query.`);
      console.error(`Please run the indexing process first.`);
      throw new Error(`Index directory ${persistDir} not found.`);
    }
    console.error("Error checking persistence directory:", error);
    throw error; // Re-throw unexpected errors
  }

  try {
    // Load the storage context from the existing directory
    const storageContext: StorageContext = await storageContextFromDefaults({
      persistDir: persistDir,
    });

    // Load the index using the storage context
    console.log("Loading index from storage...");
    // Initialize the index object; the actual data is loaded via the storage context
    const index: VectorStoreIndex = await VectorStoreIndex.init({
        storageContext: storageContext,
        // Add serviceContext if needed (e.g., specific LLM or embedding model for querying)
        // serviceContext: serviceContext // example
    });
    console.log("Index loaded successfully.");

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


async function performVectorSearchOnly(
    queryText: string,
    persistDir: string
  ) {
    console.log("Starting vector search...");
    console.log(`Persistence directory: ${persistDir}`);
    console.log(`Query: "${queryText}"`);

    try {
        // Check if the persistence directory exists. Querying requires it.
        await fs.stat(persistDir);
    } catch (error: any) {
        if (error.code === 'ENOENT') {
        console.error(`Error: Persistence directory '${persistDir}' not found. Cannot query.`);
        console.error(`Please run the indexing process first.`);
        throw new Error(`Index directory ${persistDir} not found.`);
        }
        console.error("Error checking persistence directory:", error);
        throw error; // Re-throw unexpected errors
    }

    try {

        // 1. --- Load the Index from Disk ---
        // Configure storage context to load from the directory
        const storageContext = await storageContextFromDefaults({
            persistDir: persistDir,
        });

        console.log(`Loading index from ${persistDir}...`);
        const index: VectorStoreIndex = await VectorStoreIndex.init({
            storageContext: storageContext,
            // Add serviceContext if needed (e.g., specific LLM or embedding model for querying)
            // serviceContext: serviceContext // example
        });
        console.log("Index loaded successfully.");


        // 2. --- Get the Retriever from the Index ---
        // This is the key step to get direct access to the retrieval mechanism
        // You can configure the retriever, e.g., specify how many top results to return
        const retriever: BaseRetriever = index.asRetriever({
            similarityTopK: 5, // Example: Get top 5 most similar nodes
        });
        console.log("Retriever obtained from index.");


        // 3. --- Perform the Retrieval (Vector Search) ---
        console.log(`Performing retrieval for query: "${queryText}"`);
        // Use the retriever's 'retrieve' method directly
        const retrievedNodes: NodeWithScore[] = await retriever.retrieve(queryText);
        console.log("Retrieval completed.");


        // 4. --- Process the Results ---
        // The 'retrievedNodes' array contains the raw search results
        console.log(`\nFound ${retrievedNodes.length} relevant nodes:`);
        if (retrievedNodes.length > 0) {
            retrievedNodes.forEach((nodeWithScore, index) => {
            console.log(`\n--- Node ${index + 1} (Score: ${nodeWithScore.score?.toFixed(4) ?? 'N/A'}) ---`);
            // Access the text content of the node
            console.log(nodeWithScore.node.getContent(MetadataMode.NONE));
            // You can also access metadata if available:
            // console.log("Metadata:", nodeWithScore.node.metadata);
            });
        } else {
            console.log("No relevant nodes found for the query.");
        }

    } catch (error) {
      console.error("An error occurred:", error);
    }
  }

// --- Example Usage ---

async function main() {
  // Configuration
  const persistDir = "./llamaIndexLocalDataBase"; // Choose your storage directory
  const documentPath = "src/combined-all.pdf"; // Adjust if your path is different
  const userQuery = "Diamond Minimum Color: H"; // Example query

  // 1. Index the document (only if storage directory doesn't exist)
  // Make sure the documentPath is correct relative to where you run the script
  // await indexDocument(documentPath, persistDir);

  performVectorSearchOnly(userQuery, persistDir)

  console.log("\n==================================================\n"); // Separator

  // 2. Query the index
//   try {
//     const { message, sourceNodes } = await queryIndex(userQuery, persistDir);

//     // Output query results
//     console.log("\nQuery Response:");
//     console.log("Output response with sources: \n", message.content, '\n\n');

//     if (sourceNodes && sourceNodes.length > 0) {
//       console.log("\nSource Nodes:");
//       sourceNodes.forEach((source: NodeWithScore, idx: number) => {
//         console.log(
//           `${idx + 1}: Score: ${source?.score?.toFixed(3)} - "${source.node.getContent(MetadataMode.NONE).substring(0, 80)}..."\n`,
//         );
//         // console.log(`   (Node ID: ${source.node.id_})`); // Optional: log node ID
//       });
//     } else {
//       console.log("No source nodes found for this query.");
//     }
//   } catch (error) {
//     console.error("\nFailed to execute query.", error);
//     // Handle the error appropriately (e.g., inform the user the index needs creation)
//   }
}


// Run the main function
main().catch(console.error);
