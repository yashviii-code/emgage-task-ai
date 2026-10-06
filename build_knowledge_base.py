"""
Build Knowledge Base Pipeline
==============================
Runs the complete RAG pipeline:
  1. Crawl docs.emgage.work
  2. Chunk the text
  3. Generate embeddings
  4. Build FAISS vector database

Usage:
    python build_knowledge_base.py
"""

import sys
import os

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():

    print()
    print("=" * 70)
    print("EMGAGE KNOWLEDGE BASE BUILDER")
    print("=" * 70)
    print()

    # =========================================================
    # STEP 1: CRAWL
    # =========================================================

    print("STEP 1/4: Crawling docs.emgage.work ...")
    print()

    from rag.crawler import crawl_website, save_pages

    pages = crawl_website()
    save_pages(pages)

    if not pages:
        print("ERROR: No pages were crawled. Exiting.")
        sys.exit(1)

    print(f"\n✓ Crawled {len(pages)} pages\n")

    # =========================================================
    # STEP 2: CHUNK
    # =========================================================

    print("STEP 2/4: Chunking text ...")
    print()

    from rag.chunker import load_pages, create_chunks, save_chunks

    pages = load_pages()
    chunks = create_chunks(pages)
    save_chunks(chunks)

    if not chunks:
        print("ERROR: No chunks were created. Exiting.")
        sys.exit(1)

    print(f"\n✓ Created {len(chunks)} chunks\n")

    # =========================================================
    # STEP 3: EMBED
    # =========================================================

    print("STEP 3/4: Generating embeddings ...")
    print()

    from rag.embeddings import (
        load_chunks as load_chunks_for_embed,
        load_model,
        create_embeddings,
        save_embeddings,
    )

    chunks_data = load_chunks_for_embed()
    model = load_model()
    embeddings = create_embeddings(model, chunks_data)
    save_embeddings(chunks_data, embeddings)

    print(f"\n✓ Generated {len(embeddings)} embeddings\n")

    # =========================================================
    # STEP 4: BUILD VECTOR DB (ChromaDB)
    # =========================================================

    print("STEP 4/4: Building ChromaDB vector database ...")
    print()

    from rag.vectordb import (
        load_embeddings,
        build_chroma_db,
    )

    data = load_embeddings()
    collection = build_chroma_db(data)

    print()
    print("=" * 70)
    print("✓ KNOWLEDGE BASE BUILT SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("You can now run the app:")
    print("  streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
