"""Source the raw data for the F1 Race Finder.

Two sources are combined:

1. Jolpica F1 API (https://github.com/jolpica/jolpica-f1), the community-run
   successor to the Ergast API. Gives structured facts for every World
   Championship race: season, round, date, circuit, country and the podium.
   It also gives the Wikipedia URL of each race, which is how the two sources
   are joined.
2. The Wikipedia API. Gives the article for each race (fetched as wikitext in
   batches of 50 and converted to plain text), which holds the narrative
   (background, qualifying, race report, aftermath) we search over.

Outputs:
    data/races.csv          one row per race (structured metadata)
    data/articles.jsonl     one JSON object per race: {race_id, title, text}

Usage:
    python scripts/fetch_data.py               # all seasons 1950 -> current
    python scripts/fetch_data.py --start 2020  # a subset, handy for testing
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.parse
from pathlib import Path

import mwparserfromhell
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOLPICA = "https://api.jolpi.ca/ergast/f1"
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API clients to send a descriptive User-Agent.
HEADERS = {"User-Agent": "F1RaceFinder/1.0 (student course project; python-requests)"}

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url: str, params: dict | None = None, retries: int = 8) -> dict:
    """GET with simple exponential backoff (both APIs rate-limit)."""
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2 ** attempt * 2, 60)
            print(f"  {resp.status_code} from {url}; retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Gave up on {url}")


# ---------------------------------------------------------------- Jolpica ---

def fetch_position(season: int, position: int) -> list[dict]:
    """All races in a season, each with the single result at `position`."""
    data = get_json(f"{JOLPICA}/{season}/results/{position}.json", {"limit": 100})
    time.sleep(0.35)  # Jolpica: max 4 req/s burst, 500 req/hour
    return data["MRData"]["RaceTable"]["Races"]


def driver_name(result: dict) -> str:
    d = result["Driver"]
    return f"{d['givenName']} {d['familyName']}"


def fetch_races(start: int, end: int) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for season in range(start, end + 1):
        print(f"Season {season}")
        for pos in (1, 2, 3):
            for race in fetch_position(season, pos):
                race_id = f"{race['season']}-{int(race['round']):02d}"
                loc = race["Circuit"]["Location"]
                row = rows.setdefault(race_id, {
                    "race_id": race_id,
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "date": race["date"],
                    "circuit": race["Circuit"]["circuitName"],
                    "locality": loc.get("locality", ""),
                    "country": loc.get("country", ""),
                    "wiki_url": race["url"],
                })
                # Shared drives (1950s) list several drivers at one position;
                # keep the first, which is what most record books credit.
                result = race["Results"][0]
                key = {1: "winner", 2: "second", 3: "third"}[pos]
                row.setdefault(key, driver_name(result))
                if pos == 1:
                    row.setdefault("winner_team", result["Constructor"]["name"])
                    row.setdefault("winner_grid", int(result.get("grid", 0) or 0))
    return pd.DataFrame(rows.values()).sort_values(["season", "round"])


# -------------------------------------------------------------- Wikipedia ---

def wiki_title(url: str) -> str:
    return urllib.parse.unquote(url.rsplit("/wiki/", 1)[1]).replace("_", " ")


BATCH = 50  # the API returns full content for up to 50 pages per request


def wikitext_to_text(wikitext: str) -> str:
    """Convert wikitext to plain text, keeping '== Heading ==' lines for chunking."""
    code = mwparserfromhell.parse(wikitext)
    # Remove citations, images/categories and tables before stripping markup.
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).lower() in {"ref", "table", "gallery", "math"}:
            try:
                code.remove(tag)
            except ValueError:
                pass
    for link in code.filter_wikilinks(recursive=False):
        if str(link.title).strip().lower().startswith(("file:", "image:", "category:")):
            code.remove(link)

    out = []
    for section in code.get_sections(flat=True, include_lead=True):
        headings = section.filter_headings(recursive=False)
        if headings:
            h = headings[0]
            marks = "=" * h.level
            out.append(f"\n{marks} {h.title.strip_code().strip()} {marks}")
            section.remove(h)
        out.append(section.strip_code(normalize=True, collapse=True).strip())
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_batch(titles: list[str]) -> dict[str, dict]:
    """Fetch up to 50 articles in one call. Returns {requested_title: article}."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "redirects": 1,
        "titles": "|".join(titles),
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
    }
    query = get_json(WIKI_API, params)["query"]
    # Follow title normalisation and redirects back to what we asked for.
    alias = {t: t for t in titles}
    for step in ("normalized", "redirects"):
        for m in query.get(step, []):
            for k, v in alias.items():
                if v == m["from"]:
                    alias[k] = m["to"]
    pages = {p["title"]: p for p in query["pages"]}
    out = {}
    for requested, final in alias.items():
        page = pages.get(final, {})
        revs = page.get("revisions")
        wikitext = revs[0]["slots"]["main"]["content"] if revs else ""
        out[requested] = {"title": final, "text": wikitext_to_text(wikitext)}
    return out


def fetch_articles(races: pd.DataFrame, out_path: Path) -> None:
    # Resume support: skip races we already have.
    done: set[str] = set()
    if out_path.exists():
        done = {json.loads(line)["race_id"] for line in out_path.open()}
    todo = races[~races["race_id"].isin(done)]
    print(f"Fetching {len(todo)} Wikipedia articles ({len(done)} already cached)")

    with out_path.open("a") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo.iloc[i:i + BATCH]
            titles = [wiki_title(u) for u in batch.wiki_url]
            articles = fetch_batch(titles)
            for row, title in zip(batch.itertuples(), titles):
                art = articles[title]
                if not art["text"]:
                    # A few API links are wrong; fall back to "<season> <race name>".
                    fallback = f"{row.season} {row.race_name}"
                    art = fetch_batch([fallback])[fallback]
                    print(f"  {row.race_id}: '{title}' missing, tried '{fallback}': "
                          f"{'ok' if art['text'] else 'still empty'}")
                f.write(json.dumps({"race_id": row.race_id, **art}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")
            time.sleep(3)  # be polite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=1950)
    parser.add_argument("--end", type=int, default=dt.date.today().year)
    parser.add_argument("--skip-races", action="store_true",
                        help="reuse data/races.csv and only fetch missing articles")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if args.skip_races:
        races = pd.read_csv(DATA_DIR / "races.csv")
    else:
        races = fetch_races(args.start, args.end)
        races.to_csv(DATA_DIR / "races.csv", index=False)
        print(f"Saved {len(races)} races to data/races.csv")

    fetch_articles(races, DATA_DIR / "articles.jsonl")
    print("Done.")


if __name__ == "__main__":
    main()
