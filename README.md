# Emgage Docs AI

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant for Emgage documentation. The app searches a local ChromaDB vector database, retrieves the most relevant documentation chunks, and answers user questions using a Groq-hosted LLM while grounding the response in source material and uploaded PDF content.

## Overview

This project is designed to help users ask natural-language questions about Emgage documentation and internal PDFs without needing to manually browse large knowledge bases. It combines:

- Web crawling of Emgage documentation
- Text cleaning and chunking
- Embedding generation
- Vector search with ChromaDB
- PDF ingestion for custom documents
- A Streamlit chat interface
- Groq LLM responses grounded in retrieved context

## Key Features

- Semantic search across official Emgage docs
- PDF upload support for custom knowledge sources
- ChromaDB-backed retrieval layer
- Direct Groq API integration for LLM responses
- Source-aware answer formatting
- Dark-themed Streamlit interface
- Built-in knowledge base creation pipeline

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Groq API
- Requests
- PyPDF
- BeautifulSoup
- LangChain Text Splitter

## Project Structure

```text
Emgage task/
├── app.py                     # Streamlit application
├── build_knowledge_base.py    # End-to-end indexing pipeline
├── requirements.txt           # Python dependencies
├── .env                       # Local environment variables (not committed)
├── .streamlit/                # Streamlit config
├── data/
│   ├── crawled/              # Crawled docs, chunks, embeddings
│   └── chroma_db/            # Persistent ChromaDB vector store
├── rag/
│   ├── cleaner.py            # Text cleaning utilities
│   ├── chunker.py            # Chunking logic
│   ├── crawler.py            # Website crawling logic
│   ├── embeddings.py         # Embedding generation
│   ├── retriever.py          # Retrieval logic with ChromaDB
│   ├── vectordb.py           # Vector DB creation/search helpers
│   └── __init__.py
└── .gitignore
```

## Prerequisites

- Python 3.10+
- A Groq API key
- Access to the internet for crawling documentation
- A local environment with enough disk space for embeddings and ChromaDB persistence

## Setup

1. Clone or open the repository in your working directory.
2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the project root with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Building the Knowledge Base

Before running the app, build the local documentation index:

```bash
python build_knowledge_base.py
```

This script runs the full pipeline:

1. Crawls Emgage documentation pages
2. Saves raw page content
3. Cleans and chunks the text
4. Generates embeddings
5. Stores the vectors in ChromaDB

If the index is missing or empty, the app will show that the knowledge base is unavailable or not loaded.

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal in your browser.

## Using the App

Once the app is running:

- Ask questions about Emgage documentation.
- Upload a PDF from the sidebar to index and query it alongside the official docs.
- Review grounded responses with source references when available.
- Use the app as a Q&A assistant for HRMS-related documentation.

## How It Works

The application follows this flow:

1. User enters a question in the Streamlit chat interface.
2. The app searches the ChromaDB index using the query embedding.
3. Relevant document chunks are retrieved.
4. The retrieved context is sent to the Groq LLM as a system prompt.
5. The model answers using the retrieved content and adds source references when relevant.

## Data Storage

The project stores generated data under the `data/` directory:

- `data/crawled/` contains crawled pages, chunked text, and generated embeddings
- `data/chroma_db/` contains the persistent ChromaDB vector database

## Troubleshooting

### `GROQ_API_KEY not configured in .env file.`

Create or update the `.env` file in the project root and include a valid `GROQ_API_KEY`.

### `No relevant documentation found.`

This usually means the knowledge base has not been built yet or the ChromaDB collection is empty. Run:

```bash
python build_knowledge_base.py
```

### ChromaDB path or Windows compatibility issues

The project includes a safe ChromaDB path helper to handle Windows path issues, but if the vector store is corrupted or stale, you may need to clear the folder under `data/chroma_db` and rebuild the index.

## Notes

- The app is optimized for factual, source-grounded answers.
- The default model uses Groq and falls back to a secondary model if needed.
- Uploaded PDFs are embedded and indexed into the same vector store so they can be queried alongside the main documentation.

## License

This project is intended for internal or project-specific use and does not include a separate public license file unless explicitly added by the repository owner.
