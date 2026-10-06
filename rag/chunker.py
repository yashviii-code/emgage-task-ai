import json
import os
import sys

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

try:
    from .cleaner import clean_text
except ImportError:
    from cleaner import clean_text


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "data", "crawled", "emgage_docs.json")

OUTPUT_FILE = os.path.join(BASE_DIR, "data", "crawled", "emgage_chunks.json")

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


# ============================================================
# LOAD CRAWLED DATA
# ============================================================

def load_pages():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Could not find: {INPUT_FILE}\n"
            "Run crawler.py first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        pages = json.load(file)

    return pages


# ============================================================
# CREATE TEXT SPLITTER
# ============================================================

def create_splitter():

    return RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    )


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(pages):

    splitter = create_splitter()

    chunks = []

    for page_number, page in enumerate(pages):

        url = page.get(
            "url",
            ""
        )

        title = page.get(
            "title",
            ""
        )

        raw_text = page.get(
            "text",
            ""
        )

        # Clean webpage text
        text = clean_text(
            raw_text
        )

        if not text:

            print(
                f"Skipping empty page: {url}"
            )

            continue

        # Split page
        page_chunks = splitter.split_text(
            text
        )

        print(
            f"\nPage {page_number + 1}: "
            f"{title}"
        )

        print(
            f"URL: {url}"
        )

        print(
            f"Chunks created: "
            f"{len(page_chunks)}"
        )

        # Store each chunk
        for chunk_number, chunk in enumerate(
            page_chunks
        ):

            chunk_data = {

                "chunk_id": (
                    f"page_{page_number + 1}"
                    f"_chunk_{chunk_number + 1}"
                ),

                "content": chunk,

                "metadata": {

                    "source": url,

                    "title": title,

                    "page_number": (
                        page_number + 1
                    ),

                    "chunk_number": (
                        chunk_number + 1
                    ),

                }

            }

            chunks.append(
                chunk_data
            )

    return chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(chunks):

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("EMGAGE DOCUMENTATION CHUNKER")
    print("=" * 70)

    print(
        f"Input: {INPUT_FILE}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    # Load pages
    pages = load_pages()

    print(
        f"\nPages loaded: {len(pages)}"
    )

    # Create chunks
    chunks = create_chunks(
        pages
    )

    # Save
    save_chunks(
        chunks
    )

    print(
        "\nChunking completed successfully."
    )