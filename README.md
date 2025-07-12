# RAG (Retrieval-Augmented Generation) Service

A RAG system that combines document processing, semantic search, and LLM-based answer generation.

## Installation

1. Set up your Hugging Face token (required for model access) using one of these methods:
   ```bash
   # Option 1: Export as environment variable
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   
   # Option 2: Login via CLI
   huggingface-cli login
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## High Level Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingestion     │───▶│    Indexing     │───▶│   Retrieval     │───▶│   Generation    │
│ (Load Documents)│    │ (Create FAISS)  │    │ (Find Relevant) │    │ (LLM Response)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ DocumentProcessor│    │  VectorStore    │    │ SentenceTransf. │    │ AutoModelForGen │
│ Excel/Word Files │    │ FAISS + Metadata│    │ Embeddings      │    │ LLM Inference   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Technical Stack Overview

The RAG implementation leverages these key technologies:

- **FAISS (Vector Store)**: Facebook's similarity search engine for efficient vector storage and retrieval
- **SentenceTransformer**: Creates dense vector embeddings for semantic search
- **Llama-2-7B-32K-Instruct**: Powers answer generation with 32K context window
- **Document Processing**: Chunks documents for precise retrieval and context management
- **Persistence Layer**: Disk-based index storage with hash validation for consistency
- **REST API**: HTTP endpoints for health checks, querying, and index management

The system follows a pipeline architecture:
1. Document ingestion (Excel/Word)
2. Semantic indexing
3. Context retrieval
4. LLM-based answer generation

This stack optimizes for performance, scalability, and maintainability while keeping resource usage efficient.

## System Flow

### Indexing Phase
1. Place your documents in the `data/` directory
   - Supported formats: Excel (.xlsx, .xls) and Word (.docx)
2. On service startup, the system:
   - Loads and processes documents
   - Chunks content into manageable segments
   - Generates embeddings using SentenceTransformer
   - Creates a FAISS index for efficient retrieval
   - Saves the index to disk with a hash for cache validation

### Query Phase
1. Send a POST request to `/query` with your question
2. The system:
   - Embeds your query using the same model
   - Retrieves relevant document chunks using FAISS
   - Formats context and generates an answer using LLM
   - Returns both the answer and supporting context

## API Usage

### 1. Start the Service
```bash
python rag_service.py
```
The service runs on `http://localhost:8000` by default.

### 2. Available Endpoints

- **Health Check**
  ```bash
  GET /health
  ```

- **Index Status**
  ```bash
  GET /index-status
  ```

- **Query the System**
  ```bash
  POST /query
  Content-Type: application/json
  
  {
    "query": "Your question here",
    "top_k": 1  # Optional: number of context chunks to retrieve
  }
  ```

- **Rebuild Index**
  ```bash
  POST /rebuild-index
  ```

### 3. Example Usage
You can use the provided `client.py` to test the system:
```bash
python client.py
```

## System Components

- **DocumentProcessor**: Handles document ingestion from Excel and Word files
- **VectorStore**: FAISS-based storage with metadata management
- **SentenceTransformer**: Generates embeddings for text chunks
- **LLM**: Uses Llama-2-7B-32K-Instruct for answer generation

## Logs Example (First time running the service)

```
INFO:rag_service:Starting RAG service...
INFO:rag_service:Building new index...
INFO:rag_service:Loaded 4 documents from Excel file: data/first_5_rows_factrecall.xlsx
INFO:rag_service:Loaded Word document: data/Factrecall.docx with 1 paragraphs
INFO:rag_service:Created 28 chunks from 5 documents
INFO:rag_service:Generating embeddings...
INFO:rag_service:Index built successfully with 28 chunks
```

## Client Input & Output Logs

Input Query: "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?"

```
=== Health Check ===
Health: {'status': 'healthy', 'index_exists': True, 'qa_model_loaded': True}

=== Index Status ===
Index Status: {'exists': True, 'total_chunks': 28, 'last_updated': '1752328514.930704'}

=== Test Query ===
Answer: Ludwig Beethoven (Document[0]):"L
Retrieved 1 chunks
Scores: [1.3338711261749268]

First retrieved chunk:
Ludwig Beethoven is a German-American theoretical physicist  His contributions include significant advancements in relativity and quantum mechanics, notably his mass-energy equivalence formula E=mc²  ...
```
