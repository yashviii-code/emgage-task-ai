import json
import os

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "data", "crawled", "emgage_chunks.json")

OUTPUT_FILE = os.path.join(BASE_DIR, "data", "crawled", "emgage_embeddings.json")

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"\nCould not find: {INPUT_FILE}\n"
            "Please run chunker.py first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:

        raise ValueError(
            "No chunks found in the input file."
        )

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print()
    print("=" * 70)
    print("LOADING E5 EMBEDDING MODEL")
    print("=" * 70)

    print(
        f"Model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded successfully."
    )

    return model


# ============================================================
# CREATE DOCUMENT EMBEDDINGS
# ============================================================

def create_embeddings(
    model,
    chunks
):

    print()
    print("=" * 70)
    print("CREATING DOCUMENT EMBEDDINGS")
    print("=" * 70)

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    print(
        f"Number of chunks: {len(texts)}"
    )

    print(
        "Generating embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print()
    print(
        "Embedding generation completed."
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    chunks,
    embeddings
):

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    output_data = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        output_data.append({

            "chunk_id": chunk[
                "chunk_id"
            ],

            "content": chunk[
                "content"
            ],

            "metadata": chunk[
                "metadata"
            ],

            "embedding": embedding.tolist()

        })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            ensure_ascii=False
        )

    print()
    print("=" * 70)

    print(
        f"Embeddings saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"Total vectors: "
        f"{len(output_data)}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EMGAGE RAG - E5 EMBEDDING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print()
    print(
        f"Chunks loaded: {len(chunks)}"
    )

    # --------------------------------------------------------
    # STEP 2: Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # STEP 3: Generate embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        model,
        chunks
    )

    # --------------------------------------------------------
    # STEP 4: Save embeddings
    # --------------------------------------------------------

    save_embeddings(
        chunks,
        embeddings
    )

    print()
    print(
        "Embedding pipeline completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

