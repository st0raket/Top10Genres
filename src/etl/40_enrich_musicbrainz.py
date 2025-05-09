"""
40_enrich_musicbrainz.py

Enrich unique_tracks.csv with MusicBrainz & AcousticBrainz features:
- extract(): load tracks with mbid
- transform(): fetch AB features with retry/backoff
- load(): write acousticbrainz_api_enriched_tracks.csv
"""
import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests_cache
from tqdm import tqdm
from dotenv import load_dotenv

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))


import config

# ─── Configuration ─────────────────────────────────────────────────────────
load_dotenv()
# assume unique_tracks_with_mbid.csv exists

session = requests_cache.CachedSession(cache_name='ab_cache', backend='filesystem', expire_after=86400)
session.headers.update({'User-Agent': os.getenv('AB_USER_AGENT', 'CapstoneApp/1.0')})
MAX_WORKERS = int(os.getenv('AB_WORKERS', 1))
THROTTLE = float(os.getenv('AB_THROTTLE', 1.0))
RETRIES = int(os.getenv('AB_RETRIES', 3))

# ─── Fetch helper ────────────────────────────────────────────────────────────
def fetch_ab(mbid: str) -> dict:
    url = f"https://acousticbrainz.org/{mbid}/high-level"
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=15)
            if not getattr(r, 'from_cache', False): time.sleep(THROTTLE)
            r.raise_for_status()
            data = r.json().get('highlevel', {}) or {}
            rec = {'mbid': mbid}
            for k, v in data.items():
                if isinstance(v, dict) and 'value' in v and 'probability' in v:
                    key = k.replace('-', '_')
                    rec[f'ab_{key}_value'] = v['value']
                    rec[f'ab_{key}_prob'] = v['probability']
                elif isinstance(v, (int, float)):
                    rec[f'ab_{k}'] = v
            return rec
        except Exception as e:
            logging.warning(f"AB fetch error for {mbid} (attempt {attempt+1}): {e}")
    return {'mbid': mbid, 'error': 'failed'}

# ─── Pipeline Functions ──────────────────────────────────────────────────────

def extract(input_csv: Path) -> list[str]:
    df = pd.read_csv(input_csv)
    return df['mbid'].dropna().astype(str).tolist()


def transform(mbids: list[str]) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(fetch_ab, mbid): mbid for mbid in mbids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="AB enrichment"):
            results.append(fut.result())
    return results


def load(data: list[dict], output_csv: Path) -> None:
    pd.DataFrame(data).to_csv(output_csv, index=False)
    logging.info(f"AB features written to {output_csv}")

# ─── CLI & Main ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    mbid_list = extract(config.PROCESSED_DIR / 'unique_tracks_with_mbid.csv')
    enriched = transform(mbid_list)
    load(enriched, config.PROCESSED_DIR / 'acousticbrainz_api_enriched_tracks.csv')


if __name__ == '__main__':
    main()
