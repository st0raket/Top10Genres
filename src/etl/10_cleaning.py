# src/etl/10_cleaning.py
"""
10_cleaning.py

Pipeline step: Clean and aggregate Spotify chart data.
- extract(): read raw Spotify CSV
- transform(): clean/tag, aggregate Spotify data
- load(): write aggregated Spotify table and unique tracks list
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import re

import sys
# Ensure the project root is on the sys.path
# This allows 'import config' to work correctly
# Assuming this script is in src/etl/ and config.py is in the project root
try:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
except NameError: # Fallback for interactive environments
    root = Path.cwd() # Adjust if your CWD is not the project root
    if not (root / 'config.py').exists(): # A simple check
        # Try to find project root assuming a common structure like 'project_root/src/etl'
        for i in range(4):
            if (root / 'config.py').exists():
                break
            root = root.parent
        if not (root / 'config.py').exists():
            raise ImportError("Could not find config.py. Ensure project root is correctly identified.")
    sys.path.insert(0, str(root))


import config

# ─── Cleaning Helpers ──────────────────────────────────────────────────────────
# Regex for cleaning artist and song titles
_FEAT_DROP_RE = re.compile(r"\s+\(??feat\..*?$", flags=re.I) # Remove "feat." and following text
_PAREN_DROP_RE = re.compile(r"\s*\[.*?]|\(.*?\)$") # Remove text in brackets or parentheses at the end
_QUOTE_PAIR_RE = re.compile(r'""') # Replace double quotes with single
_QUOTE_EDGE_RE = re.compile(r"^[\'\"“”‘’]+|[\'\"“”‘’]+$") # Remove quotes at the start/end
_SPLIT_SPOTIFY_RE = re.compile(r"\s*,\s*", flags=re.U) # Split multiple artists in Spotify data

def tidy(text: str) -> str:
    """
    Cleans a string by removing featured artists, parenthetical remarks,
    and standardizing quotes and whitespace.
    """
    s = str(text)
    for rx in (_FEAT_DROP_RE, _PAREN_DROP_RE):
        s = rx.sub("", s)
    s = _QUOTE_PAIR_RE.sub('"', s)
    s = _QUOTE_EDGE_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def primary_artist_spotify(raw: str) -> str:
    """
    Extracts the primary artist from a raw artist string (Spotify specific).
    Assumes primary artist is the first one listed if multiple.
    """
    raw = raw.strip()
    return _SPLIT_SPOTIFY_RE.split(raw, 1)[0].strip()

# ─── Pipeline Functions ──────────────────────────────────────────────────────────

def extract(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load raw Spotify CSV into a DataFrame.
    """
    spotify_path = raw_dir / 'spotify_charts_data_raw.csv'
    data = {}
    if not spotify_path.exists():
        logging.error(f"Spotify raw data file not found: {spotify_path}")
        raise FileNotFoundError(f"Spotify raw data file not found: {spotify_path}")

    df_spotify = pd.read_csv(spotify_path)
    data['spotify'] = df_spotify
    logging.info(f"Loaded {len(df_spotify)} rows from {spotify_path.name}")
    return data


def transform(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean/tag and aggregate Spotify data.
    Returns spotify_weekly (aggregated data) and unique_tracks.
    """
    df_spotify = data['spotify']

    # Clean Spotify data
    date_col = 'Date' # Assuming 'Date' is the column name in spotify_charts_data_raw.csv
    # Ensure 'Date' column exists
    if date_col not in df_spotify.columns:
        logging.error(f"'{date_col}' column not found in Spotify data.")
        # Attempt to find a date-like column if common alternatives are used
        # This is a basic fallback, specific logic might be needed if column names vary significantly
        potential_date_cols = [col for col in df_spotify.columns if 'date' in col.lower()]
        if potential_date_cols:
            date_col = potential_date_cols[0]
            logging.warning(f"Using '{date_col}' as date column for Spotify based on keyword match.")
        else:
            raise KeyError(f"Date column not found in Spotify data. Checked for '{date_col}' and similar.")


    df_spotify[date_col] = pd.to_datetime(df_spotify[date_col].astype(str), errors='coerce')
    df_spotify = df_spotify.dropna(subset=[date_col]).copy() # Use .copy() to avoid SettingWithCopyWarning
    df_spotify['date'] = df_spotify[date_col].dt.strftime('%Y-%m-%d')

    # Ensure 'Artists' and 'Song Title' columns exist for Spotify
    spotify_artist_col = 'Artists' # As per original script
    spotify_song_col = 'Song Title' # As per original script

    if spotify_artist_col not in df_spotify.columns:
        logging.error(f"'{spotify_artist_col}' column not found in Spotify data.")
        raise KeyError(f"'{spotify_artist_col}' column not found in Spotify data.")
    if spotify_song_col not in df_spotify.columns:
        logging.error(f"'{spotify_song_col}' column not found in Spotify data.")
        raise KeyError(f"'{spotify_song_col}' column not found in Spotify data.")

    df_spotify['artist'] = df_spotify[spotify_artist_col].map(tidy).map(primary_artist_spotify)
    df_spotify['song'] = df_spotify[spotify_song_col].map(tidy)

    df_spotify = df_spotify[(df_spotify['artist'] != '') & (df_spotify['song'] != '')]
    df_spotify['year_week'] = pd.to_datetime(df_spotify['date']).dt.to_period('W').astype(str)
    logging.info(f"Cleaned Spotify data: {len(df_spotify)} rows")

    # Aggregate Spotify data
    # Ensure 'Peak Position' is numeric before aggregation if it exists
    spotify_peak_pos_col = 'Peak Position' # As per original script
    if spotify_peak_pos_col in df_spotify.columns:
        df_spotify[spotify_peak_pos_col] = pd.to_numeric(df_spotify[spotify_peak_pos_col], errors='coerce')
    
    numeric_spotify = df_spotify.select_dtypes('number').columns
    # Ensure 'Position' (or similar rank column) is included if it's numeric
    spotify_rank_col = 'Position' # Common name for rank in Spotify charts
    if spotify_rank_col in df_spotify.columns and pd.api.types.is_numeric_dtype(df_spotify[spotify_rank_col]) and spotify_rank_col not in numeric_spotify:
        numeric_spotify = numeric_spotify.tolist() + [spotify_rank_col]


    if not numeric_spotify.empty:
        spotify_w = df_spotify.groupby(['year_week', 'artist', 'song'], as_index=False)[numeric_spotify].mean()
        spotify_w = spotify_w.rename(columns={c: f"spotify_{c.lower().replace(' ', '_')}" for c in numeric_spotify}) # Standardize column names
    else:
        # If no numeric columns, group by keys and maybe count occurrences or handle as error
        logging.warning("No numeric columns found in Spotify data for aggregation. Creating table with keys only.")
        spotify_w = df_spotify[['year_week', 'artist', 'song']].drop_duplicates().reset_index(drop=True)
        # Add a count column if desired
        # spotify_w_counts = df_spotify.groupby(['year_week', 'artist', 'song']).size().reset_index(name='spotify_occurrences')
        # spotify_w = spotify_w.merge(spotify_w_counts, on=['year_week', 'artist', 'song'], how='left')


    logging.info(f"Aggregated Spotify data: {len(spotify_w)} rows")

    # `merged_weekly.csv` will now be just the Spotify weekly data.
    # The unique tracks will be derived from this Spotify data.
    unique_tracks = spotify_w[['artist', 'song']].drop_duplicates().reset_index(drop=True)
    logging.info(f"Derived {len(unique_tracks)} unique tracks from Spotify data.")

    return spotify_w, unique_tracks


def load(
    spotify_w: pd.DataFrame,
    unique_tracks: pd.DataFrame,
    out_dir: Path
) -> None:
    """
    Write aggregated Spotify table and unique track list.
    """
    out_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # Save the aggregated Spotify data as 'merged_weekly.csv' for consistency with downstream scripts
    merged_weekly_path = out_dir / 'merged_weekly.csv'
    spotify_w.to_csv(merged_weekly_path, index=False)
    logging.info(f"Written aggregated Spotify data (as merged_weekly.csv) to {merged_weekly_path}")

    # Save unique tracks (now only from Spotify)
    # The `50_merge_features.py` script expects `unique_tracks.csv` in `DATA_DIR/working`
    # Adjust if your config points `config.DATA_DIR` elsewhere or if `50_merge_features.py` is changed
    working_dir = config.DATA_DIR / 'working'
    working_dir.mkdir(parents=True, exist_ok=True)
    unique_tracks_path = working_dir / 'unique_tracks.csv'
    unique_tracks.to_csv(unique_tracks_path, index=False)
    logging.info(f"Wrote {len(unique_tracks):,} unique tracks to {unique_tracks_path}")

# ─── CLI & Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and aggregate Spotify chart data"
    )
    parser.add_argument(
        '--raw-dir', type=Path,
        default=config.RAW_DIR, # Uses RAW_DIR from your config.py
        help='Directory containing raw CSVs (specifically spotify_charts_data_raw.csv)'
    )
    parser.add_argument(
        '--out-dir', type=Path,
        default=config.PROCESSED_DIR, # Uses PROCESSED_DIR from your config.py
        help="Directory to write processed CSVs (merged_weekly.csv)"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    args = parse_args()
    raw_data = extract(args.raw_dir)
    spotify_weekly_data, unique_tracks_data = transform(raw_data)
    load(spotify_weekly_data, unique_tracks_data, args.out_dir)
    logging.info('✅ Spotify cleaning and aggregation pipeline completed successfully')


if __name__ == '__main__':
    main()
