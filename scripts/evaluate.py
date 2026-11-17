"""Evaluate retrieval quality and compare embedding models.

Each test query describes a famous race *without naming it* (mostly), the way
a fan who half-remembers it would. We check where the correct race lands in
the ranked race list.

Metrics (race-level, after collapsing chunks to races):
    Hit@1, Hit@5 - share of queries whose correct race is ranked 1st / top 5
    MRR          - mean reciprocal rank (1/rank, 0 if not in top 20)

Usage:
    python scripts/evaluate.py                       # compare default models
    python scripts/evaluate.py --models BAAI/bge-small-en-v1.5
    python scripts/evaluate.py --rerank   # the deployed pipeline: Chroma + cross-encoder
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from f1_search import (COLLECTION_NAME, EMBED_MODEL, RERANK_CANDIDATES,  # noqa: E402
                       RERANK_MODEL, embed_text, query_text, rerank)
from build_index import load_chunks  # noqa: E402

# (query, season, race name as in races.csv)
EVAL_SET = [
    ("Lauda withdrew because the rain was too dangerous and Hunt won the title by a single point",
     1976, "Japanese Grand Prix"),
    ("championship rival driven off the road at the first corner to clinch the title",
     1990, "Japanese Grand Prix"),
    ("teammates collided at the chicane and one was later disqualified, handing the other the championship",
     1989, "Japanese Grand Prix"),
    ("overtook a struggling car on slick tyres in the final corners to win the championship by one point",
     2008, "Brazilian Grand Prix"),
    ("controversial safety car restart on the final lap decided the world championship",
     2021, "Abu Dhabi Grand Prix"),
    ("only six cars started after a tyre safety dispute and the crowd booed",
     2005, "United States Grand Prix"),
    ("weekend where a three-time world champion was killed at Tamburello",
     1994, "San Marino Grand Prix"),
    ("won from last place after a four hour race interrupted by rain, passing the leader on the final lap",
     2011, "Canadian Grand Prix"),
    ("brand new team that bought Honda's operation won on its debut",
     2009, "Australian Grand Prix"),
    ("only three cars finished after a wet race of attrition and a Ligier won",
     1996, "Monaco Grand Prix"),
    ("race run entirely behind the safety car in the rain and only half points were awarded",
     2021, "Belgian Grand Prix"),
    ("car split in two and burst into flames after hitting the barrier, driver escaped the fire",
     2020, "Bahrain Grand Prix"),
    ("Multi 21 team order ignored by Vettel",
     2013, "Malaysian Grand Prix"),
    ("Barrichello told to give the win to Schumacher on the finishing line, crowd booed on the podium",
     2002, "Austrian Grand Prix"),
    ("wheel-banging duel for second place between Villeneuve and Arnoux in the last laps",
     1979, "French Grand Prix"),
    ("Williams' surprise win followed by a garage fire",
     2012, "Spanish Grand Prix"),
    ("driver crashed deliberately to bring out the safety car and help his teammate win at night",
     2008, "Singapore Grand Prix"),
    ("18-year-old became the youngest ever winner in his first race after being promoted to the senior team",
     2016, "Spanish Grand Prix"),
    ("car aquaplaned off in the wet and hit a recovery crane, driver later died of his injuries",
     2014, "Japanese Grand Prix"),
    ("left rear tyre exploded on the straight at high speed, ending his title hopes",
     1986, "Australian Grand Prix"),
    ("fiery crash at the Nordschleife, driver pulled from the burning car by other drivers",
     1976, "German Grand Prix"),
    ("shock maiden win for AlphaTauri after a red flag and a penalty for the leader",
     2020, "Italian Grand Prix"),
    ("title decider where one driver turned into his rival and was stripped of his championship runner-up place",
     1997, "European Grand Prix"),
    ("huge multi-car pileup at the first corner at Spa in torrential rain",
     1998, "Belgian Grand Prix"),
    ("first race of the season where Mansell won on debut for Ferrari with the semi-automatic gearbox",
     1989, "Brazilian Grand Prix"),
]

DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # the starter notebook's model
    "BAAI/bge-small-en-v1.5",
]


def race_ids(eval_set) -> list[str]:
    races = pd.read_csv(ROOT / "data" / "races.csv")
    ids = []
    for _, season, name in eval_set:
        match = races[(races.season == season) & (races.race_name == name)]
        assert len(match) == 1, f"no unique race for {season} {name}"
        ids.append(match.race_id.iloc[0])
    return ids


def evaluate(model_name: str, chunks, targets) -> dict:
    model = SentenceTransformer(model_name)
    t0 = time.time()
    doc_emb = model.encode([embed_text(c) for c in chunks], batch_size=64,
                           normalize_embeddings=True, show_progress_bar=True)
    embed_time = time.time() - t0
    q_emb = model.encode([query_text(q, model_name) for q, *_ in EVAL_SET],
                         normalize_embeddings=True)
    chunk_race = np.array([c.race_id for c in chunks])

    ranks = []
    for qi, target in enumerate(targets):
        scores = doc_emb @ q_emb[qi]
        ranked_races = list(dict.fromkeys(chunk_race[np.argsort(-scores)[:500]]))
        rank = ranked_races.index(target) + 1 if target in ranked_races[:20] else None
        ranks.append(rank)
    rr = [1 / r if r else 0 for r in ranks]
    return {
        "model": model_name,
        "hit@1": np.mean([r == 1 for r in ranks]),
        "hit@5": np.mean([bool(r) and r <= 5 for r in ranks]),
        "mrr": np.mean(rr),
        "embed_s": embed_time,
        "ranks": ranks,
    }


def summarize(ranks: list) -> str:
    hit1 = np.mean([r == 1 for r in ranks])
    hit5 = np.mean([bool(r) and r <= 5 for r in ranks])
    mrr = np.mean([1 / r if r else 0 for r in ranks])
    return f"Hit@1 {hit1:.0%}  Hit@5 {hit5:.0%}  MRR {mrr:.2f}"


def evaluate_rerank(targets) -> None:
    """Evaluate the built chroma_db with and without the cross-encoder re-ranker."""
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import CrossEncoder

    model = SentenceTransformer(EMBED_MODEL)
    ce = CrossEncoder(RERANK_MODEL)
    col = chromadb.PersistentClient(path=str(ROOT / "chroma_db"), settings=Settings(
        anonymized_telemetry=False)).get_collection(COLLECTION_NAME)

    def rank_of(target, metas):
        order = list(dict.fromkeys(m["race_id"] for m in metas))[:20]
        return order.index(target) + 1 if target in order else None

    before, after = [], []
    for (q, *_), target in zip(EVAL_SET, targets):
        res = col.query(query_embeddings=model.encode([query_text(q)], normalize_embeddings=True).tolist(),
                        n_results=RERANK_CANDIDATES, include=["metadatas", "documents"])
        metas, docs = res["metadatas"][0], res["documents"][0]
        before.append(rank_of(target, metas))
        _, metas2, _ = rerank(ce, q, metas, docs)
        after.append(rank_of(target, metas2))
    print(f"Embedding only:      {summarize(before)}")
    print(f"+ cross-encoder:     {summarize(after)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--rerank", action="store_true",
                        help="evaluate the built chroma_db with the re-ranker")
    args = parser.parse_args()

    if args.rerank:
        evaluate_rerank(race_ids(EVAL_SET))
        return

    chunks = load_chunks()
    targets = race_ids(EVAL_SET)
    results = [evaluate(m, chunks, targets) for m in args.models]

    print("\n| Model | Hit@1 | Hit@5 | MRR | Embed time |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| `{r['model']}` | {r['hit@1']:.0%} | {r['hit@5']:.0%} | "
              f"{r['mrr']:.2f} | {r['embed_s']:.0f}s |")
    print("\nPer-query rank of the correct race (- = not in top 20):")
    for i, (q, season, name) in enumerate(EVAL_SET):
        cells = "  ".join(f"{(r['ranks'][i] or '-')!s:>2}" for r in results)
        print(f"{cells}   {season} {name}: {q[:60]}")


if __name__ == "__main__":
    main()
