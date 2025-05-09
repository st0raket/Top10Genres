"""
20_enrich_deezer.py

Enrich unique_tracks.csv with Deezer metadata:
- extract(): load unique tracks
- transform(): fetch Deezer features with retry
- load(): write unique_tracks_deezer_enriched.csv
"""
import argparse
import csv
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))


import config
# ─── HTTP Session ───────────────────────────────────────────────────────────
session = requests.Session()
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount('https://', adapter)
session.mount('http://', adapter)

# ─── Pipeline Functions ──────────────────────────────────────────────────────

def extract(input_csv: Path) -> list[dict]:
    """Load unique_tracks.csv into list of dict"""
    with input_csv.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def transform(rows: list[dict], max_workers: int, max_retries: int) -> list[dict]:
    """Fetch Deezer features, with retry logic"""
    def get_features(song: str, artist: str) -> dict | None:
        # same logic as original: search, fetch track and album
        try:
            params = {"q": f'track:"{song}" artist:"{artist}"'}
            resp = session.get("https://api.deezer.com/search", params=params, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json().get("data", [])
            if not data:
                return None
            t = data[0]
            features = {"song": song, "artist": artist}
            track_id = t.get("id")
            if track_id:
                r2 = session.get(f"https://api.deezer.com/track/{track_id}", timeout=10)
                if r2.status_code == 200:
                    ti = r2.json()
                    features.update({
                        "track_id": track_id,
                        "duration_sec": ti.get("duration"),
                        "explicit_lyrics": ti.get("explicit_lyrics"),
                        "bpm": ti.get("bpm"),
                        "gain": ti.get("gain"),
                    })
                album = t.get("album", {})
                album_id = album.get("id")
                if album_id:
                    r3 = session.get(f"https://api.deezer.com/album/{album_id}", timeout=10)
                    if r3.status_code == 200:
                        a = r3.json()
                        features.update({
                            "album_id": album_id,
                            "release_date": a.get("release_date"),
                            "genres": ",".join(g.get("name") for g in a.get("genres", {}).get("data", [])),
                            "album_track_count": a.get("nb_tracks"),
                            "album_record_type": a.get("record_type"),
                            "album_duration_sec": a.get("duration"),
                            "album_explicit_lyrics": a.get("explicit_lyrics"),
                        })
            return features
        except Exception as e:
            logging.warning(f"Error fetching Deezer for {artist} - {song}: {e}")
            return None

    enriched = {}
    # load existing if any
    output_csv = config.PROCESSED_DIR / 'unique_tracks_deezer_enriched.csv'
    if output_csv.exists():
        with output_csv.open('r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                enriched[(row['song'], row['artist'])] = row
        logging.info(f"Loaded {len(enriched)} existing Deezer records")

    for attempt in range(1, max_retries+1):
        tasks = [r for r in rows if (r['song'], r['artist']) not in enriched]
        if not tasks:
            break
        logging.info(f"Attempt {attempt}: fetching {len(tasks)} records")
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for feat in tqdm(exe.map(lambda x: get_features(x['song'], x['artist']), tasks), total=len(tasks), desc=f"Deezer pass {attempt}"):
                if feat:
                    enriched[(feat['song'], feat['artist'])] = feat

    return list(enriched.values())


def load(data: list[dict], output_csv: Path) -> None:
    """Write enriched list to CSV"""
    if not data:
        logging.error("No Deezer data to write")
        return
    fieldnames = list(data[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logging.info(f"Deezer enrichment complete: {len(data)} rows → {output_csv}")


# ─── CLI & Main ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Enrich tracks with Deezer metadata")
    parser.add_argument('--max-workers', type=int, default=5)
    parser.add_argument('--max-retries', type=int, default=3)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    args = parse_args()
    input_csv = config.DATA_DIR / 'working' / 'unique_tracks.csv'
    rows = extract(input_csv)
    enriched = transform(rows, args.max_workers, args.max_retries)
    load(enriched, config.PROCESSED_DIR / 'unique_tracks_deezer_enriched.csv')


if __name__ == '__main__':
    main()