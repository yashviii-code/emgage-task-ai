import json
import os
import sys
import chromadb
from chromadb.config import Settings

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "crawled", "emgage_embeddings.json")

CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

COLLECTION_NAME = "emgage_docs"


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


def get_chroma_client(persist_dir: str = None) -> chromadb.PersistentClient:
    """Get or create persistent ChromaDB client."""
    safe_dir = get_safe_chroma_path(persist_dir or CHROMA_DIR)
    return chromadb.PersistentClient(path=safe_dir)


def get_chroma_collection(client: chromadb.PersistentClient = None, name: str = COLLECTION_NAME):
    """Get or create the ChromaDB collection."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"\nCould not find: {EMBEDDINGS_FILE}\n\n"
            "Run embeddings.py first."
        )

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not data:
        raise ValueError("The embeddings file is empty.")

    return data


# ============================================================
# POPULATE CHROMA DATABASE
# ============================================================

def build_chroma_db(data, persist_dir: str = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
    print()
    print("=" * 70)
    print("CREATING CHROMADB VECTOR DATABASE")
    print("=" * 70)

    client = get_chroma_client(persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for item in data:
        ids.append(str(item["chunk_id"]))
        documents.append(item["content"])
        
        # Flatten metadata dict for ChromaDB compatibility
        meta = {}
        for k, v in item.get("metadata", {}).items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)
        metadatas.append(meta)
        embeddings.append(item["embedding"])

    # Batch upsert into ChromaDB
    batch_size = 100
    total = len(ids)
    print(f"Upserting {total} items into ChromaDB collection '{collection_name}'...")

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            embeddings=embeddings[i:end]
        )

    count = collection.count()
    print(f"Total documents stored in ChromaDB: {count}")
    return collection


# ============================================================
# TEST SEARCH
# ============================================================

def test_search(collection, data, query_index=0, top_k=3):
    print()
    print("=" * 70)
    print("TESTING CHROMADB VECTOR SEARCH")
    print("=" * 70)

    query_embedding = data[query_index]["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print(f"Test chunk: {data[query_index]['chunk_id']}")
    print()

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        # In cosine distance: score = 1 - distance
        score = 1.0 - dist
        print(f"Result {rank}")
        print(f"Score: {score:.4f}")
        print(f"Title: {meta.get('title', '')}")
        print(f"Source: {meta.get('source', '')}")
        print(f"Content preview: {doc[:200]}...")
        print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("EMGAGE RAG - CHROMADB VECTOR DATABASE")
    print("=" * 70)

    data = load_embeddings()
    print(f"\nLoaded {len(data)} embedded chunks.")

    collection = build_chroma_db(data)
    test_search(collection, data)

    print()
    print("=" * 70)
    print("CHROMADB DATABASE CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Chroma persistence path: {CHROMA_DIR}")
    print()


if __name__ == "__main__":
    main()