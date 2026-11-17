"""F1 Race Finder - semantic search over every Formula One World Championship race.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from f1_search import COLLECTION_NAME, EMBED_MODEL, best_per_race, build_where, query_text

ROOT = Path(__file__).resolve().parent
CANDIDATE_CHUNKS = 150  # chunks pulled from Chroma before collapsing to races

EXAMPLES = [
    "Title decided on the last lap in the rain",
    "Driver refused to race because conditions were too dangerous",
    "Team orders scandal that the crowd booed",
    "Underdog team's shock first win",
    "Huge first-corner pileup",
    "Car caught fire and the driver escaped",
    "Rookie wins in his first race for a new team",
    "Race stopped with a red flag and restarted",
]

FLAGS = {
    "UK": "🇬🇧", "USA": "🇺🇸", "United States": "🇺🇸", "Italy": "🇮🇹", "Monaco": "🇲🇨",
    "Germany": "🇩🇪", "France": "🇫🇷", "Belgium": "🇧🇪", "Netherlands": "🇳🇱",
    "Spain": "🇪🇸", "Portugal": "🇵🇹", "Switzerland": "🇨🇭", "Austria": "🇦🇹",
    "Hungary": "🇭🇺", "Sweden": "🇸🇪", "Argentina": "🇦🇷", "Brazil": "🇧🇷",
    "Mexico": "🇲🇽", "Canada": "🇨🇦", "South Africa": "🇿🇦", "Morocco": "🇲🇦",
    "Japan": "🇯🇵", "China": "🇨🇳", "Korea": "🇰🇷", "Malaysia": "🇲🇾",
    "Singapore": "🇸🇬", "India": "🇮🇳", "Australia": "🇦🇺", "Bahrain": "🇧🇭",
    "UAE": "🇦🇪", "Abu Dhabi": "🇦🇪", "Turkey": "🇹🇷", "Russia": "🇷🇺",
    "Azerbaijan": "🇦🇿", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Vietnam": "🇻🇳",
}

st.set_page_config(page_title="F1 Race Finder", page_icon="🏁", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; max-width: 1100px; }
    .hero h1 { font-size: 2.6rem; margin-bottom: 0; letter-spacing: -0.02em; }
    .hero h1 span { color: #E10600; }
    .hero p { color: #A8A8B3; font-size: 1.08rem; margin-top: .35rem; }
    .race-card {
        border: 1px solid rgba(255,255,255,.09); border-left: 4px solid #E10600;
        border-radius: 10px; padding: 1.05rem 1.25rem 1rem; margin-bottom: .9rem;
        background: rgba(255,255,255,.025);
    }
    .race-card .top { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
    .race-card h3 { margin: 0; font-size: 1.25rem; }
    .race-card .rank { color: #E10600; font-weight: 700; margin-right: .4rem; }
    .race-card .meta { color: #A8A8B3; font-size: .9rem; margin: .2rem 0 .6rem; }
    .race-card .podium { font-size: .92rem; margin-bottom: .65rem; }
    .race-card .podium b { font-weight: 600; }
    .race-card .section { display: inline-block; font-size: .72rem; text-transform: uppercase;
        letter-spacing: .06em; color: #E10600; border: 1px solid rgba(225,6,0,.45);
        border-radius: 4px; padding: .05rem .4rem; margin-bottom: .4rem; }
    .race-card .passage { font-size: .95rem; line-height: 1.55; color: #E6E6EA; }
    .race-card .foot { margin-top: .6rem; font-size: .85rem; }
    .race-card .foot a { color: #FF4D4D; text-decoration: none; }
    .score { font-variant-numeric: tabular-nums; color: #A8A8B3; font-size: .85rem; white-space: nowrap; }
    .score .bar { display: inline-block; width: 70px; height: 6px; border-radius: 3px;
        background: rgba(255,255,255,.12); vertical-align: middle; margin-left: .4rem; overflow: hidden; }
    .score .bar i { display: block; height: 100%; background: #E10600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------- resources ---

@st.cache_resource(show_spinner="Loading the embedding model...")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Opening the vector database...")
def load_collection():
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"),
                                       settings=Settings(anonymized_telemetry=False))
    return client.get_collection(COLLECTION_NAME, embedding_function=None)


@st.cache_data
def load_races() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "races.csv").set_index("race_id", drop=False)


def search(query: str, k: int, season_range, winner, team) -> list[dict]:
    q_emb = load_model().encode([query_text(query)], normalize_embeddings=True).tolist()
    res = load_collection().query(
        query_embeddings=q_emb,
        n_results=CANDIDATE_CHUNKS,
        where=build_where(season_range, winner, team),
        include=["distances", "metadatas", "documents"],
    )
    return best_per_race(res["ids"][0], res["distances"][0], res["metadatas"][0],
                         res["documents"][0], k)


# ------------------------------------------------------------------ UI ---

races = load_races()
first, last = int(races.season.min()), int(races.season.max())

st.markdown(
    f"""<div class="hero"><h1>🏁 F1 Race <span>Finder</span></h1>
    <p>Describe a race you half-remember. Search all {len(races):,} Formula One
    World Championship Grands Prix ({first}–{last}) by <i>meaning</i>, not keywords.</p></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filters")
    season_range = st.slider("Seasons", first, last, (first, last))
    winner = st.selectbox("Race winner", ["Any"] + sorted(races.winner.dropna().unique()))
    team = st.selectbox("Winning team", ["Any"] + sorted(races.winner_team.dropna().unique()))
    k = st.slider("Number of results", 3, 20, 8)
    st.divider()
    st.caption(
        "Race reports from Wikipedia (CC BY-SA 4.0); results data from the "
        "[Jolpica F1 API](https://github.com/jolpica/jolpica-f1). Embeddings: "
        f"`{EMBED_MODEL}`. Vector store: Chroma."
    )

if "query" not in st.session_state:
    st.session_state.query = ""


def use_example(text: str) -> None:
    st.session_state.query = text


query = st.text_input(
    "Search",
    key="query",
    placeholder="e.g. wet race where the championship leader pulled out because it was too dangerous",
    label_visibility="collapsed",
)

st.caption("Try one:")
cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    cols[i % 4].button(ex, key=f"ex{i}", on_click=use_example, args=(ex,), use_container_width=True)

if query.strip():
    with st.spinner("Searching..."):
        hits = search(
            query.strip(), k,
            season_range if season_range != (first, last) else None,
            None if winner == "Any" else winner,
            None if team == "Any" else team,
        )
    if not hits:
        st.info("No races match those filters. Try widening the season range.")
    st.write("")
    for rank, hit in enumerate(hits, 1):
        race = races.loc[hit["race_id"]]
        flag = FLAGS.get(race.country, "🏁")
        date = pd.to_datetime(race.date).strftime("%-d %B %Y")
        podium = ""
        if isinstance(race.winner, str):
            parts = [f"🥇 <b>{html.escape(race.winner)}</b> ({html.escape(str(race.winner_team))})"]
            for medal, col in (("🥈", "second"), ("🥉", "third")):
                if isinstance(race[col], str):
                    parts.append(f"{medal} {html.escape(race[col])}")
            podium = " &nbsp;·&nbsp; ".join(parts)
        pct = max(0.0, min(1.0, hit["score"]))
        extra = (f" &nbsp;·&nbsp; {hit['also_matched']} more matching passage"
                 f"{'s' if hit['also_matched'] != 1 else ''}") if hit["also_matched"] else ""
        st.markdown(
            f"""<div class="race-card">
              <div class="top">
                <h3><span class="rank">{rank}</span>{race.season} {html.escape(race.race_name)}</h3>
                <span class="score">match {pct:.2f}<span class="bar"><i style="width:{pct*100:.0f}%"></i></span></span>
              </div>
              <div class="meta">{flag} {html.escape(race.circuit)}, {html.escape(str(race.locality))} &nbsp;·&nbsp; {date} &nbsp;·&nbsp; Round {race['round']}</div>
              <div class="podium">{podium}</div>
              <div class="section">{html.escape(hit['section'])}</div>
              <div class="passage">{html.escape(hit['passage'])}</div>
              <div class="foot"><a href="{race.wiki_url}" target="_blank">Read the full race report on Wikipedia →</a>{extra}</div>
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.write("")
    with st.expander("How does this work?", expanded=True):
        st.markdown(
            """
            1. **Data sourcing.** Race facts (date, circuit, podium) come from the Jolpica F1 API;
               the race reports come from each race's Wikipedia article.
            2. **Chunking.** Articles are split along their own sections (Background, Qualifying,
               Race, ...) into passages of up to ~180 words, each prefixed with the race name so
               the passage knows which race it belongs to.
            3. **Embedding.** Every passage becomes a 384-dimensional vector with
               `BAAI/bge-small-en-v1.5`, chosen over the starter model after an evaluation on 25
               famous-race queries (see the README).
            4. **Vector DB.** The ~13.7k vectors live in a persistent **Chroma** collection.
            5. **Search.** Your query is embedded the same way, Chroma finds the nearest passages
               (cosine similarity), and results are grouped so you see each race once, with the
               passage that matched best.
            """
        )
