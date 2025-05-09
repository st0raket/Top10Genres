"""
30_enrich_lastfm.py

Enrich unique_tracks.csv with Last.fm metadata.
- extract(): load unique tracks
- transform(): fetch metadata in parallel with caching
- load(): write lastfm_enriched_tracks.csv
"""
import os
import time
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests_cache
from tqdm import tqdm
import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))


import config

import config
import config
from dotenv import load_dotenv

# ─── Configuration ─────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("LASTFM_API_KEY") or os.getenv("LASTFM_API")
if not API_KEY:
    raise RuntimeError("Set LASTFM_API_KEY in environment or .env")

# cache
session = requests_cache.CachedSession(cache_name="lastfm_cache", backend="filesystem", expire_after=86400)

# ─── HTTP / Parsing Helpers ──────────────────────────────────────────────────
def call_lastfm(method: str, params: dict) -> dict:
    payload = {**params, "method": method, "api_key": API_KEY, "format": "json"}
    resp = session.get("https://ws.audioscrobbler.com/2.0/", params=payload, timeout=15)
    if not getattr(resp, 'from_cache', False): time.sleep(0.1)
    try:
        resp.raise_for_status()
    except Exception:
        logging.warning(f"Last.fm HTTP error {resp.status_code}")
        return {}
    return resp.json()

def parse_info(row: pd.Series, info: dict) -> dict:
    tags = info.get("toptags", {}).get("tag", [])
    tag_list = [t.get("name") for t in tags if isinstance(t, dict)]
    return {
        "artist": row.artist,
        "song": row.song,
        "listeners": int(info.get("listeners", 0)),
        "playcount": int(info.get("playcount", 0)),
        "duration_ms": int(info.get("duration", 0)),
        "album": (info.get("album") or {}).get("title"),
        "tags": "|".join(tag_list)[:255],
        "error": None
    }

# ─── Pipeline Functions ──────────────────────────────────────────────────────

def extract(input_csv: Path) -> pd.DataFrame:
    return pd.read_csv(input_csv)


def transform(df: pd.DataFrame, workers: int) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(lambda r: call_lastfm("track.getInfo", {"artist": r.artist, "track": r.song}), row): row for row in df.itertuples(index=False)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Last.fm enrichment"):
            info = fut.result().get("track") or {}
            results.append(parse_info(futures[fut], info))
    return results


def load(data: list[dict], output_csv: Path) -> None:
    pd.DataFrame(data).to_csv(output_csv, index=False)
    logging.info(f"Last.fm enrichment written to {output_csv}")

# ─── CLI & Main ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Enrich tracks with Last.fm data")
    parser.add_argument('--workers', type=int, default=5)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    args = parse_args()
    tracks = extract(config.PROCESSED_DIR / 'unique_tracks.csv')
    enriched = transform(tracks, args.workers)
    load(enriched, config.PROCESSED_DIR / 'lastfm_enriched_tracks.csv')


if __name__ == '__main__':
    main()