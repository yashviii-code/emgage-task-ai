import os
import sys
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

COLLECTION_NAME = "emgage_docs"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


def get_safe_chroma_path(path: str = CHROMA_DIR) -> str:
    """Resolve Windows 8.3 short path to avoid non-ASCII / Unicode path issues in native C++/Rust vector DB engines."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(500)
            ret = ctypes.windll.kernel32.GetShortPathNameW(abs_path, buf, 500)
            if ret > 0 and buf.value:
                return buf.value
        except Exception:
            pass
    return abs_path


# ============================================================
# RETRIEVER CLASS
# ============================================================

class EmgageRetriever:
    """
    Retrieves relevant Emgage documentation and uploaded PDF chunks
    using ChromaDB vector search with all-MiniLM-L6-v2 embeddings.
    """

    def __init__(self, chroma_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.model: Optional[SentenceTransformer] = None
        self._loaded = False

    def load(self):
        """Load ChromaDB client, collection, and all-MiniLM-L6-v2 embedding model."""
        if self._loaded:
            return

        print("Initializing ChromaDB client...")
        safe_path = get_safe_chroma_path(self.chroma_dir)
        self.client = chromadb.PersistentClient(path=safe_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        total_docs = self.collection.count()
        print(f"ChromaDB collection '{self.collection_name}' loaded: {total_docs} items.")

        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")

        self._loaded = True

    def embed_query(self, query: str) -> List[float]:
        """Embed user query using all-MiniLM-L6-v2."""
        if not self._loaded:
            self.load()
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Search for most relevant chunks given a user query.
        Returns list of dicts with content, source, title, score.
        """
        if not self._loaded:
            self.load()

        if self.collection.count() == 0:
            return []

        query_vector = self.embed_query(query)
        actual_k = min(top_k, self.collection.count())
        if actual_k <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if not results["documents"] or not results["documents"][0]:
            return output

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            meta = meta or {}
            score = 1.0 - dist  # Cosine similarity
            output.append({
                "content": doc,
                "source": meta.get("source", ""),
                "title": meta.get("title", meta.get("source", "Documentation")),
                "score": float(score),
                "is_pdf": meta.get("is_pdf", False)
            })

        return output

    def add_pdf_content(self, filename: str, text_chunks: List[str]):
        """
        Dynamically chunk, embed, and index uploaded PDF content into ChromaDB.
        """
        if not self._loaded:
            self.load()

        if not text_chunks:
            return

        embeddings = self.model.encode(text_chunks, normalize_embeddings=True)
        ids = [f"pdf_{filename}_{i}" for i in range(len(text_chunks))]
        metadatas = [
            {
                "source": filename,
                "title": f"PDF: {filename}",
                "page_chunk": i + 1,
                "is_pdf": True
            }
            for i in range(len(text_chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=text_chunks,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )
        print(f"Indexed {len(text_chunks)} chunks from PDF '{filename}' into ChromaDB.")

    def get_context(self, query: str, top_k: int = TOP_K) -> str:
        """
        Retrieve relevant chunks and format as a context string for LLM.
        """
        results = self.search(query, top_k)
        if not results:
            return "No relevant documentation found."

        context_parts = []
        for i, res in enumerate(results, 1):
            source_label = res.get("title") or res.get("source") or f"Doc {i}"
            url_line = f"URL / Source: {res.get('source')}\n" if res.get("source") else ""
            context_parts.append(
                f"--- Source {i}: {source_label} ---\n"
                f"{url_line}"
                f"{res['content']}\n"
            )

        return "\n".join(context_parts)


if __name__ == "__main__":
    print("Testing EmgageRetriever with ChromaDB...")
    retriever = EmgageRetriever()
    retriever.load()
    sample_queries = [
        "How does HR letter setup work?",
        "How to process attendance?",
        "What is Emgage HRMS?"
    ]
    for q in sample_queries:
        print(f"\nQuery: {q}")
        hits = retriever.search(q, top_k=2)
        for h in hits:
            print(f"  → [{h['score']:.3f}] {h['title']} ({h['source']})")
