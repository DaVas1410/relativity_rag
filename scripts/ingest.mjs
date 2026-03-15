import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import * as fs from "fs";
import * as path from "path";

// Define the paths
const DATA_DIR = path.join(process.cwd(), "data");
const VECTOR_STORE_DIR = path.join(process.cwd(), "vector_store");

async function ingest() {
  console.log("Starting PDF ingestion...");
  
  // Ensure the data directory exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR);
    console.log("Created 'data' directory. Please drop your General Relativity PDFs there and run again.");
    return;
  }

  // Get all PDF files
  const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith(".pdf"));
  if (files.length === 0) {
    console.log("No PDF files found in the 'data' directory. Add some and try again.");
    return;
  }

  const allDocs = [];

  // Load and split PDFs
  for (const file of files) {
    const filePath = path.join(DATA_DIR, file);
    console.log(`Loading ${file}...`);
    
    const loader = new PDFLoader(filePath);
    const rawDocs = await loader.load();
    
    // Split the text into manageable chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    
    const docs = await textSplitter.splitDocuments(rawDocs);
    allDocs.push(...docs);
    console.log(`Loaded ${docs.length} chunks from ${file}`);
  }

  console.log(`Total chunks: ${allDocs.length}`);
  
  // Create embeddings using local Ollama model (e.g., nomic-embed-text)
  // Ensure you have run: ollama run nomic-embed-text 
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", // A lightweight, excellent embedding model
    baseUrl: "http://localhost:11434",
  });

  console.log("Generating embeddings and building vector store. This may take a while...");
  const vectorStore = await HNSWLib.fromDocuments(allDocs, embeddings);
  
  // Save the vector store locally
  if (!fs.existsSync(VECTOR_STORE_DIR)) {
    fs.mkdirSync(VECTOR_STORE_DIR);
  }
  
  await vectorStore.save(VECTOR_STORE_DIR);
  console.log(`Vector store saved successfully to ${VECTOR_STORE_DIR}!`);
}

ingest().catch(console.error);