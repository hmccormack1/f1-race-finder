"""Chunk the race articles, embed them and store them in a persistent Chroma DB.

Usage:
    python scripts/build_index.py                      # default model -> chroma_db/
    python scripts/build_index.py --model sentence-transformers/all-MiniLM-L6-v2

See f1_search.py for the chunking strategy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from f1_search import COLLECTION_NAME, EMBED_MODEL, chunk_article, embed_text  # noqa: E402


def load_chunks() -> list:
    races = pd.read_csv(ROOT / "data" / "races.csv")
    articles = {}
    for line in (ROOT / "data" / "articles.jsonl").open():
        art = json.loads(line)
        articles[art["race_id"]] = art
    chunks = []
    for _, race in races.iterrows():
        # A race whose article could not be fetched still gets its "Race facts" chunk.
        art = articles.get(race.race_id) or {"title": f"{race.season} {race.race_name}", "text": ""}
        chunks += chunk_article(race, art["title"], art["text"])
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=EMBED_MODEL)
    parser.add_argument("--out", default=str(ROOT / "chroma_db"))
    args = parser.parse_args()

    chunks = load_chunks()
    words = [len(c.text.split()) for c in chunks]
    print(f"{len(chunks)} chunks from {len({c.race_id for c in chunks})} races "
          f"(mean {sum(words) / len(words):.0f} words, max {max(words)})")

    model = SentenceTransformer(args.model)
    t0 = time.time()
    embeddings = model.encode(
        [embed_text(c) for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors -> cosine similarity
    )
    print(f"Embedded in {time.time() - t0:.0f}s, dim={embeddings.shape[1]}")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    client = chromadb.PersistentClient(path=str(out), settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model},
        embedding_function=None,  # we always pass our own vectors
    )
    step = 2000  # Chroma has a max batch size
    for i in range(0, len(chunks), step):
        batch = chunks[i:i + step]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embeddings[i:i + step].tolist(),
        )
    print(f"Stored {collection.count()} vectors in {out}")


if __name__ == "__main__":
    main()
