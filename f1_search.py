"""Core logic shared by the index builder, the evaluation script and the app.

Chunking strategy
-----------------
Wikipedia race articles have a consistent structure (Background, Qualifying,
Race, Post-race, ...). We chunk *along that structure*:

* Split the article on its "== Heading ==" markers so a chunk never mixes,
  say, the qualifying story with the race story.
* Drop sections that are tables or boilerplate in plain-text form
  (classification, standings, references, external links, ...).
* Pack whole paragraphs into chunks of at most ~MAX_WORDS words. A paragraph
  longer than that is split on sentence boundaries. Consecutive chunks of the
  same section share one sentence of overlap so an event described across a
  boundary is still retrievable.
* Prefix every chunk with "<race title> - <section>:" before embedding. A bare
  paragraph like "He retired on lap 12" carries no information about *which*
  race it is; the prefix puts that context into the vector.
* Add one synthetic "Race facts" chunk per race built from the structured
  data (winner, team, podium, circuit, grid). This makes factual queries such
  as "Alonso win from 14th on the grid" work even when the article is short.

Search is two-stage: Chroma returns the 100 nearest chunks by embedding
similarity, a cross-encoder re-scores those 100 against the query, and the
results are collapsed to *races*, keeping the best chunk per race.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

MAX_WORDS = 180
COLLECTION_NAME = "f1_races"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Second-stage re-ranker: reads (query, passage) together, so it is far more
# accurate than comparing two separate vectors, but too slow to run on all
# 13k passages. We only run it on the top candidates from Chroma.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_CANDIDATES = 100
# bge models were trained with this instruction prepended to *queries*
# (not documents) for retrieval.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

SKIP_SECTION = re.compile(
    r"references|external links|see also|further reading|bibliography|footnotes|"
    r"sources|citations|classification|standings|results|entry list|entrants|"
    r"championship points|^notes$|explanatory notes",
    re.IGNORECASE,
)
HEADING = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


@dataclass
class Chunk:
    race_id: str
    section: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.race_id}#{self.chunk_index}"


# ------------------------------------------------------------- chunking ---

def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_path, body) pairs. The lead has section "Summary"."""
    sections: list[tuple[str, list[str]]] = [("Summary", [])]
    path: list[str] = []
    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m:
            level = len(m.group(1)) - 2  # "==" -> 0, "===" -> 1
            path = path[:level] + [m.group(2)]
            sections.append((" > ".join(path), []))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(lines).strip()) for name, lines in sections]


def clean_paragraphs(body: str) -> list[str]:
    paras = []
    for p in re.split(r"\n\s*\n|\n", body):
        p = p.strip()
        # Footnotes ("^...") and very short leftovers of tables/lists.
        if not p or p.startswith("^") or len(p.split()) < 6:
            continue
        paras.append(p)
    return paras


def split_long(paragraph: str) -> list[str]:
    """Split one paragraph into pieces of <= MAX_WORDS on sentence boundaries."""
    if len(paragraph.split()) <= MAX_WORDS:
        return [paragraph]
    pieces, current = [], []
    for sent in SENTENCE_SPLIT.split(paragraph):
        if current and len(" ".join(current + [sent]).split()) > MAX_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.append(sent)
    if current:
        pieces.append(" ".join(current))
    return pieces


def last_sentence(text: str) -> str:
    sents = SENTENCE_SPLIT.split(text)
    return sents[-1] if len(sents) > 1 else ""


def chunk_section(body: str) -> list[str]:
    units = [u for p in clean_paragraphs(body) for u in split_long(p)]
    chunks, current = [], []
    for unit in units:
        if current and len(" ".join(current + [unit]).split()) > MAX_WORDS:
            chunks.append(" ".join(current))
            overlap = last_sentence(current[-1])
            current = [overlap] if overlap else []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def race_facts(race: pd.Series) -> str:
    parts = [
        f"The {race.season} {race.race_name} was round {race['round']} of the "
        f"{race.season} Formula One season, held on {race.date} at {race.circuit} "
        f"in {race.locality}, {race.country}."
    ]
    if isinstance(race.winner, str):
        grid = int(race.winner_grid) if pd.notna(race.winner_grid) else 0
        grid_txt = "from pole position" if grid == 1 else (
            f"from {ordinal(grid)} on the grid" if grid > 1 else "")
        parts.append(f"{race.winner} won for {race.winner_team} {grid_txt}.".replace(" .", "."))
        podium = [p for p in (race.second, race.third) if isinstance(p, str)]
        if podium:
            parts.append(f"The podium was completed by {' and '.join(podium)}.")
    return " ".join(parts)


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def chunk_article(race: pd.Series, title: str, text: str) -> list[Chunk]:
    base_meta = {
        "race_id": race.race_id,
        "season": int(race.season),
        "race_name": race.race_name,
        "country": race.country,
        "winner": race.winner if isinstance(race.winner, str) else "",
        "winner_team": race.winner_team if isinstance(race.winner_team, str) else "",
    }
    chunks = [Chunk(race.race_id, "Race facts", race_facts(race), 0, dict(base_meta))]
    for section, body in split_sections(text):
        if SKIP_SECTION.search(section):
            continue
        for piece in chunk_section(body):
            meta = {**base_meta, "section": section}
            chunks.append(Chunk(race.race_id, section, piece, len(chunks), meta))
    for c in chunks:
        c.metadata.setdefault("section", c.section)
        c.metadata["title"] = title
    return chunks


def embed_text(chunk: Chunk) -> str:
    """What actually gets embedded: the chunk plus its race/section context."""
    return f"{chunk.metadata['title']} - {chunk.section}: {chunk.text}"


def query_text(query: str, model_name: str = EMBED_MODEL) -> str:
    """bge models expect an instruction prefix on queries (not on documents)."""
    return QUERY_INSTRUCTION + query if "bge" in model_name.lower() else query


# --------------------------------------------------------------- search ---

def rerank(cross_encoder, query: str, metadatas: list[dict], documents: list[str]):
    """Re-score Chroma's candidates with a cross-encoder, best first.

    Returns (scores, metadatas, documents) sorted by score. Scores are mapped
    through a sigmoid to 0-1 so they're easier to show in the UI.
    """
    import numpy as np
    pairs = [(query, f"{m['title']} - {m.get('section', '')}: {d}")
             for m, d in zip(metadatas, documents)]
    logits = np.asarray(cross_encoder.predict(pairs))
    order = np.argsort(-logits)
    scores = 1 / (1 + np.exp(-logits[order]))
    return (scores.tolist(), [metadatas[i] for i in order], [documents[i] for i in order])


def best_per_race(scores: list[float], metadatas: list[dict],
                  documents: list[str], k: int) -> list[dict]:
    """Collapse chunk hits (sorted best first) to race hits, keeping each race's best chunk."""
    seen: dict[str, dict] = {}
    for score, meta, doc in zip(scores, metadatas, documents):
        rid = meta["race_id"]
        if rid in seen:
            seen[rid]["also_matched"] += 1
            continue
        seen[rid] = {
            "race_id": rid,
            "score": score,
            "section": meta.get("section", ""),
            "passage": doc,
            "also_matched": 0,
        }
        if len(seen) == k:
            break
    return list(seen.values())


def build_where(season_range: tuple[int, int] | None = None,
                winner: str | None = None, team: str | None = None) -> dict | None:
    clauses = []
    if season_range:
        clauses += [{"season": {"$gte": season_range[0]}},
                    {"season": {"$lte": season_range[1]}}]
    if winner:
        clauses.append({"winner": winner})
    if team:
        clauses.append({"winner_team": team})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
