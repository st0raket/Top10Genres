#!/usr/bin/env python3
"""
50_merge_features.py

Pipeline step: Merge enrichment tables (AcousticBrainz, Deezer, Last.fm)
into a master feature table.
- extract(): load base tracks and all enrichment CSVs
- transform(): merge DataFrames, process Deezer genres, process Last.fm data (including tag OHE), drop duplicates
- load(): save `master_feature_table.csv` with identical output format

One-hot encodes Deezer genres and selected Last.fm tags.
Converts 0s in specified numeric Last.fm columns to NaN.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from collections import Counter

import pandas as pd
import numpy as np # Added for np.nan

# project-root on sys.path so `import config` works
try:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
except NameError: # Fallback for interactive environments if __file__ is not defined
    root = Path.cwd()
    for i in range(4): 
        if (root / 'config.py').exists():
            break
        if root.parent == root: 
            break
        root = root.parent
    if not (root / 'config.py').exists():
        print("Warning: Could not reliably determine project_root for config import. Assuming CWD or specific relative paths.")
    sys.path.insert(0, str(root))


import config

def extract(processed_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads base tracks (unique_tracks_with_mbid.csv) and enrichment data from
    AcousticBrainz, Deezer, and Last.fm.
    """
    base_file = processed_dir / 'unique_tracks_with_mbid.csv'
    if not base_file.exists():
        logging.error(f"Required base file not found: {base_file}")
        raise FileNotFoundError(f"Required file not found: {base_file}")
    base = pd.read_csv(base_file)
    logging.info(f"Loaded base tracks: {len(base)} rows from {base_file.name}")

    ab_file = processed_dir / 'acousticbrainz_api_enriched_tracks.csv'
    ab = pd.read_csv(ab_file) if ab_file.exists() else None
    if ab is not None:
        logging.info(f"Loaded AcousticBrainz enrichment: {len(ab)} rows from {ab_file.name}")
    else:
        logging.warning(f"AcousticBrainz file not found at {ab_file}. Skipping AB merge.")

    dz_file = processed_dir / 'unique_tracks_deezer_enriched.csv'
    dz = pd.read_csv(dz_file) if dz_file.exists() else None
    if dz is not None:
        logging.info(f"Loaded Deezer enrichment: {len(dz)} rows from {dz_file.name}")
    else:
        logging.warning(f"Deezer file not found at {dz_file}. Skipping Deezer merge.")

    # Load Last.fm data
    lastfm_file = processed_dir / "lastfm_enriched_tracks.csv"
    lastfm = pd.read_csv(lastfm_file) if lastfm_file.exists() else None
    if lastfm is not None:
        logging.info(f"Loaded Last.fm enrichment: {len(lastfm)} rows from {lastfm_file.name}")
        logging.info(f"Last.fm columns: {lastfm.columns.tolist()}") 
    else:
        logging.warning(f"Last.fm file not found at {lastfm_file}. Skipping Last.fm merge.")

    return base, ab, dz, lastfm

def transform(
    base: pd.DataFrame,
    ab: Optional[pd.DataFrame],
    dz: Optional[pd.DataFrame],
    lastfm: Optional[pd.DataFrame],
    top_n_lastfm_tags: int = 25 # Number of top Last.fm tags to one-hot encode
) -> pd.DataFrame:
    df = base.copy()
    logging.info(f"Base df columns before any merge: {df.columns.tolist()}") 

    # Merge AcousticBrainz 
    if ab is not None:
        if 'mbid' in df.columns and 'mb_recording_id' in ab.columns:
            df['mbid'] = df['mbid'].astype(str)
            ab['mb_recording_id'] = ab['mb_recording_id'].astype(str)
            ab_rename = {col: f"acoustic_{col}" for col in ab.columns if col not in ['mb_recording_id', 'error']}
            if 'mb_recording_id' in ab_rename: del ab_rename['mb_recording_id']
            df = df.merge(ab.rename(columns=ab_rename), left_on='mbid', right_on='mb_recording_id', how='left')
            if 'mb_recording_id' in df.columns and 'mbid' in df.columns and 'mb_recording_id' != 'mbid':
                 df.drop(columns=['mb_recording_id'], inplace=True, errors='ignore')
            logging.info("Merged AcousticBrainz features")
        else:
            logging.warning("Skipping AcousticBrainz merge: 'mbid' or 'mb_recording_id' missing.")

    # Merge Deezer
    if dz is not None:
        if 'artist' in df.columns and 'song' in df.columns and 'artist' in dz.columns and 'song' in dz.columns:
            df['temp_artist_key'] = df['artist'].astype(str).str.lower().str.strip()
            df['temp_song_key'] = df['song'].astype(str).str.lower().str.strip()
            dz['temp_artist_key'] = dz['artist'].astype(str).str.lower().str.strip()
            dz['temp_song_key'] = dz['song'].astype(str).str.lower().str.strip()
            dz_rename = {col: f"deezer_{col}" for col in dz.columns if col not in ['artist', 'song', 'temp_artist_key', 'temp_song_key']}
            if 'artist' in dz_rename: del dz_rename['artist']
            if 'song' in dz_rename: del dz_rename['song']
            df = df.merge(dz.rename(columns=dz_rename), on=['temp_artist_key', 'temp_song_key'], how='left', suffixes=('', '_dz_drop'))
            df.drop(columns=['temp_artist_key', 'temp_song_key'], inplace=True)
            df.drop(columns=[col for col in df.columns if '_dz_drop' in col], inplace=True, errors='ignore')
            logging.info("Merged Deezer features")
            deezer_genres_col = 'deezer_genres' 
            if deezer_genres_col in df.columns:
                df[deezer_genres_col] = df[deezer_genres_col].fillna("")
                df['deezer_genre_list_temp'] = (df[deezer_genres_col].astype(str).str.split(',').map(lambda lst: [g.strip().lower() for g in lst if isinstance(g, str) and g.strip()]))
                all_deezer_genres = sorted(list(set(g for genres_list in df['deezer_genre_list_temp'] for g in genres_list)))
                if all_deezer_genres:
                    genre_cols_to_add = {}
                    for genre in all_deezer_genres:
                        safe_genre_col_name = f"deezer_genre_{genre.replace(' ', '_').replace('/', '_').replace('&','and')}"
                        genre_cols_to_add[safe_genre_col_name] = df['deezer_genre_list_temp'].map(lambda lst: int(genre in lst))
                    df = pd.concat([df, pd.DataFrame(genre_cols_to_add, index=df.index)], axis=1)
                    logging.info(f"One-hot encoded Deezer genres. Added {len(genre_cols_to_add)} columns.")
                else:
                    logging.info("No Deezer genres found to one-hot encode.")
                df.drop(columns=['deezer_genre_list_temp'], inplace=True, errors='ignore')
            else:
                logging.warning(f"Column '{deezer_genres_col}' not found for Deezer genre OHE.")
        else:
            logging.warning("Skipping Deezer merge: 'artist' or 'song' columns missing in base or Deezer data.")

    # Merge and Process Last.fm data
    if lastfm is not None:
        logging.info("Attempting to merge Last.fm data...") 
        lastfm_artist_col = 'artist' 
        lastfm_track_col = 'song'   

        df_has_artist = 'artist' in df.columns
        df_has_song = 'song' in df.columns
        lastfm_has_artist = lastfm_artist_col in lastfm.columns
        lastfm_has_song = lastfm_track_col in lastfm.columns
        logging.info(f"DEBUG Last.fm merge check: df has 'artist': {df_has_artist}, df has 'song': {df_has_song}")
        logging.info(f"DEBUG Last.fm merge check: lastfm has '{lastfm_artist_col}': {lastfm_has_artist}, lastfm has '{lastfm_track_col}': {lastfm_has_song}")

        if df_has_artist and df_has_song and lastfm_has_artist and lastfm_has_song:
            df['temp_artist_key_lastfm'] = df['artist'].astype(str).str.lower().str.strip()
            df['temp_song_key_lastfm'] = df['song'].astype(str).str.lower().str.strip()
            
            lastfm['temp_artist_key_lastfm'] = lastfm[lastfm_artist_col].astype(str).str.lower().str.strip()
            lastfm['temp_song_key_lastfm'] = lastfm[lastfm_track_col].astype(str).str.lower().str.strip()

            lastfm_cols_map = {
                'lastfm_listeners': 'lastfm_listeners',
                'lastfm_playcount': 'lastfm_playcount',
                'lastfm_duration_ms':  'lastfm_duration_ms',
                'lastfm_album': 'lastfm_album_name', 
                'lastfm_tags': 'lastfm_tags_str'    
            }
            
            cols_to_select_from_lastfm = ['temp_artist_key_lastfm', 'temp_song_key_lastfm']
            current_cols_map = {} 
            
            logging.info(f"DEBUG: Last.fm DataFrame columns available for mapping: {lastfm.columns.tolist()}") 
            for original_col_in_csv, new_prefixed_name in lastfm_cols_map.items():
                if original_col_in_csv in lastfm.columns:
                    cols_to_select_from_lastfm.append(original_col_in_csv)
                    current_cols_map[original_col_in_csv] = new_prefixed_name 
                    logging.info(f"DEBUG: Will select '{original_col_in_csv}' from Last.fm and rename to '{new_prefixed_name}'") 
                else: 
                    logging.warning(f"DEBUG: Expected Last.fm column '{original_col_in_csv}' not found in Last.fm CSV.") 
            
            if len(cols_to_select_from_lastfm) > 2: 
                lastfm_subset = lastfm[list(set(cols_to_select_from_lastfm))].copy() # Use set to ensure unique columns if original_col_in_csv was a key
                lastfm_subset.rename(columns=current_cols_map, inplace=True)
                logging.info(f"DEBUG: Last.fm subset columns after rename: {lastfm_subset.columns.tolist()}") 

                df = df.merge(
                    lastfm_subset,
                    on=['temp_artist_key_lastfm', 'temp_song_key_lastfm'],
                    how='left'
                )
                logging.info(f"Merged Last.fm features. df length after merge: {len(df)}. Columns potentially added: {list(current_cols_map.values())}") 
                logging.info(f"DEBUG: df columns after Last.fm merge: {df.columns.tolist()}") 
            else:
                logging.warning("None of the specified useful Last.fm columns (listeners, playcount, etc.) were found in the Last.fm CSV. No actual Last.fm feature columns to merge.")

            df.drop(columns=['temp_artist_key_lastfm', 'temp_song_key_lastfm'], inplace=True, errors='ignore')
            
            # MODIFIED: Convert numeric Last.fm columns and handle 0s
            numeric_lfm_cols = ['lastfm_listeners', 'lastfm_playcount', 'lastfm_duration_ms']
            for col in numeric_lfm_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce to numeric, non-numerics become NaN
                    df[col] = df[col].replace(0, np.nan) # Replace 0 with NaN
                    logging.info(f"Processed numeric Last.fm column: {col} (0s converted to NaN, non-numeric to NaN)")
                else: 
                    logging.warning(f"DEBUG: Numeric Last.fm column '{col}' not found in df for conversion.") 

            tags_col_name_processed = 'lastfm_tags_str' 
            if tags_col_name_processed in df.columns:
                logging.info(f"Processing Last.fm tags from column: {tags_col_name_processed}")
                df[tags_col_name_processed] = df[tags_col_name_processed].fillna('').astype(str)
                
                list_of_tags_series = df[tags_col_name_processed].apply(lambda x: [tag.strip().lower() for tag in x.split('|') if tag.strip()])
                
                all_tags_flat_list = [tag for sublist in list_of_tags_series for tag in sublist]
                tag_counts = Counter(all_tags_flat_list)
                
                if tag_counts:
                    top_tags = [tag for tag, count in tag_counts.most_common(top_n_lastfm_tags)]
                    logging.info(f"Top {len(top_tags)} Last.fm tags selected for OHE: {top_tags[:5]}...")

                    for tag in top_tags:
                        safe_tag_col_name = f"lfm_tag_{tag.replace(' ', '_').replace('-', '_').replace('/', '_').replace('&','and').replace('(','').replace(')','')}" 
                        df[safe_tag_col_name] = list_of_tags_series.apply(lambda x: 1 if tag in x else 0)
                    logging.info(f"One-hot encoded top {len(top_tags)} Last.fm tags.")
                else:
                    logging.info("No Last.fm tags found to one-hot encode after processing.")
                
                df.drop(columns=[tags_col_name_processed], inplace=True, errors='ignore') 
            else:
                logging.warning(f"Processed Last.fm tags column '{tags_col_name_processed}' not found after merge for OHE.")
        else:
            logging.warning(f"Skipping Last.fm merge: Key columns ('artist', 'song') missing in base or Last.fm CSV (expected '{lastfm_artist_col}', '{lastfm_track_col}').")
    else: 
        logging.info("Last.fm data (lastfm DataFrame) is None. Skipping Last.fm merge entirely.") 
    # --- End of Last.fm processing ---

    final_dedup_keys = ['artist', 'song']
    if 'mbid' in df.columns:
        df['mbid'] = df['mbid'].astype(str) 
        final_dedup_keys.append('mbid')

    if all(key_col in df.columns for key_col in final_dedup_keys):
        num_before_dedup = len(df)
        df = df.drop_duplicates(subset=final_dedup_keys, keep='first')
        logging.info(f"Dropped {num_before_dedup - len(df)} duplicate records based on {final_dedup_keys}.")
    else:
        logging.warning(f"Could not perform final deduplication as one or more key columns ({final_dedup_keys}) are missing.")

    logging.info(f"Final df columns before load: {df.columns.tolist()}") 
    return df

def load(df: pd.DataFrame, processed_dir: Path) -> None:
    out_file = processed_dir / 'master_feature_table.csv'
    df.to_csv(out_file, index=False)
    
    display_path = "UNKNOWN_PATH"
    try:
        display_path = out_file.relative_to(root)
    except ValueError: 
        display_path = out_file 
        
    print(f"✅ Master feature table saved to {display_path} ({len(df)} rows, {df.shape[1]} columns)")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge feature tables (AcousticBrainz, Deezer, Last.fm) into master feature table"
    )
    parser.add_argument(
        '--processed-dir', type=Path,
        default=config.PROCESSED_DIR,
        help='Directory containing processed CSVs and for outputting master_feature_table.csv'
    )
    parser.add_argument(
        '--top-n-lastfm-tags', type=int,
        default=25, # Default to top 25 tags
        help='Number of most frequent Last.fm tags to one-hot encode.'
    )
    return parser.parse_args()

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    args = parse_args()
    base, ab, dz, lastfm = extract(args.processed_dir)
    master = transform(base, ab, dz, lastfm, top_n_lastfm_tags=args.top_n_lastfm_tags)
    load(master, args.processed_dir)

if __name__ == '__main__':
    main()
