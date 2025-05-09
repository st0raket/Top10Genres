#!/usr/bin/env python3
"""
60_merge_timeseries.py

Pipeline step: Merge master-feature table with weekly Spotify chart data,
enrich with calendar parts, lags, rolling statistics, and release date features,
then save a model-ready dataset.

Operations
----------
extract()   – load master-feature & weekly-metrics CSVs
transform() – merge, normalise, derive chart metrics, engineer release date features, add TS features
clean()     – apply the EDL cleaning rules (zero-variance, NaNs, etc.)
load()      – write final `model_dataset_weekly.csv` to <processed_dir>
"""
import sys
from pathlib import Path
import argparse
import logging

import pandas as pd
import numpy as np

# project-root on sys.path
try:
    root = Path(__file__).resolve().parents[2]     # <project-root>
    sys.path.insert(0, str(root))                  # so `import config` works
    sys.path.insert(0, str(root / "src"))          # so `import utils.*` works
except NameError: # Fallback for interactive environments
    root = Path.cwd()
    if not (root / 'config.py').exists():
        for i in range(4):
            if (root / 'config.py').exists():
                break
            if root.parent == root:
                break
            root = root.parent
        if not (root / 'config.py').exists():
            raise ImportError("Could not find config.py. Ensure project root is correctly identified.")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))


from utils.features import add_time_series_features # Assuming this utility exists
import config

def extract(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads master feature table and weekly metrics (Spotify only)."""
    feats_path = processed_dir / "master_feature_table.csv"
    weekly_path = processed_dir / "merged_weekly.csv" # This is Spotify-only weekly data

    if not feats_path.exists():
        logging.error(f"Required master feature file not found: {feats_path}")
        raise FileNotFoundError(f"Required master feature file not found: {feats_path}")
    if not weekly_path.exists():
        logging.error(f"Required weekly metrics file not found: {weekly_path}")
        raise FileNotFoundError(f"Required weekly metrics file not found: {weekly_path}")

    feats = pd.read_csv(feats_path)
    weekly = pd.read_csv(weekly_path)

    logging.info(f"Loaded master features: {len(feats):,} rows from {feats_path.name}")
    logging.info(f"Loaded weekly Spotify metrics: {len(weekly):,} rows from {weekly_path.name}")
    return feats, weekly

def transform(feats: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weekly Spotify data with master features, normalizes columns,
    derives Spotify-specific chart metrics, engineers features from Deezer release date,
    and adds time series features.
    """
    df = weekly.merge(feats, on=["artist", "song"], how="left")

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[ ()]", "", regex=True) 
        .str.replace(r"\s+", "_", regex=True)  
    )
    logging.info("Normalised column names")

    # Handle main 'date' column (weekly observation date)
    if "year_week" in df.columns: 
        try:
            df["date"] = pd.to_datetime(
                df["year_week"].str.split("/", expand=True)[1], 
                format="%Y-%m-%d",
                errors="coerce",
            )
            df.drop(columns=["year_week"], inplace=True, errors='ignore')
            logging.info("Extracted 'date' from 'year_week'.")
        except Exception as e:
            logging.warning(f"Could not parse 'year_week': {e}. Attempting to use existing 'date' column.")
    
    if "date" not in df.columns:
        logging.error("'date' column is missing and could not be derived from 'year_week'.")
        raise KeyError("'date' column is crucial and was not found.")
        
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=['date']) 

    # --- Feature Engineering from deezer_release_date ---
    deezer_release_date_col = 'deezer_release_date' 
    if deezer_release_date_col in df.columns:
        logging.info(f"Processing '{deezer_release_date_col}' for feature engineering...")
        df[deezer_release_date_col] = pd.to_datetime(df[deezer_release_date_col], errors='coerce')
        valid_dates_mask = df['date'].notna() & df[deezer_release_date_col].notna()
        
        df.loc[valid_dates_mask, 'song_age_days'] = \
            (df.loc[valid_dates_mask, 'date'] - df.loc[valid_dates_mask, deezer_release_date_col]).dt.days
        df['song_age_days'] = df['song_age_days'].fillna(-1).astype(float).astype(int) 
        logging.info("Created 'song_age_days' feature.")

        df.loc[valid_dates_mask, 'release_year'] = df.loc[valid_dates_mask, deezer_release_date_col].dt.year
        df.loc[valid_dates_mask, 'release_month'] = df.loc[valid_dates_mask, deezer_release_date_col].dt.month
        df.loc[valid_dates_mask, 'release_dayofweek'] = df.loc[valid_dates_mask, deezer_release_date_col].dt.dayofweek
        
        df['release_year'] = df['release_year'].fillna(0).astype(int)
        df['release_month'] = df['release_month'].fillna(0).astype(int)
        df['release_dayofweek'] = df['release_dayofweek'].fillna(-1).astype(int) 
        logging.info("Created 'release_year', 'release_month', 'release_dayofweek' features.")

        if 'song_age_days' in df.columns:
            df.loc[valid_dates_mask, 'is_new_release_last_28d'] = \
                (df.loc[valid_dates_mask, 'song_age_days'] >= 0) & (df.loc[valid_dates_mask, 'song_age_days'] <= 28)
            df['is_new_release_last_28d'] = df['is_new_release_last_28d'].fillna(False).astype(int) 
            logging.info("Created 'is_new_release_last_28d' feature.")
        else:
            logging.warning("'song_age_days' was not created properly, cannot create 'is_new_release_last_28d'.")
    else:
        logging.warning(f"'{deezer_release_date_col}' not found. Skipping release date feature engineering.")
    # --- END Feature Engineering from deezer_release_date ---

    spotify_raw_cols = [
        "spotify_position", "spotify_peak_position", 
        "spotify_weeks_on_chart", "spotify_streams" 
    ]
    for col in spotify_raw_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    spotify_weeks_col_name = 'spotify_weeks_on_chart' 
    if spotify_weeks_col_name in df.columns:
        df["spotify_days_on_chart"] = df[spotify_weeks_col_name] * 7
        logging.info("Derived 'spotify_days_on_chart'.")


    if "artist" in df.columns and "song" in df.columns and "date" in df.columns:
        before_duplicates = len(df)
        df.drop_duplicates(subset=["artist", "song", "date"], inplace=True, keep='first')
        logging.info(f"Dropped {before_duplicates - len(df):,} duplicate rows based on artist, song, date.")
    else:
        logging.warning("Could not drop duplicates as 'artist', 'song', or 'date' column is missing.")

    if "spotify_streams" in df.columns and "artist" in df.columns and "song" in df.columns:
        for id_col_clean in ["artist", "song"]:
            if id_col_clean in df.columns:
                df[id_col_clean] = df[id_col_clean].fillna('').astype(str).str.strip()
            else:
                logging.warning(f"ID column '{id_col_clean}' for add_time_series_features not found.")
        df = add_time_series_features(
            df,
            date_col="date",
            metric_col="spotify_streams",
            id_cols=("artist", "song"), 
            lag_windows=(1, 2, 4, 8, 12),
            roll_windows=(4, 8, 12),
        )
        logging.info("Added calendar parts, lag & rolling stats for spotify_streams.")
    else:
        logging.warning("Skipping time series feature addition: 'spotify_streams', 'artist', or 'song' column missing.")

    ohe_album_type_col = 'deezer_album_record_type'
    if ohe_album_type_col in df.columns:
        logging.info(f"Value counts for '{ohe_album_type_col}':\n{df[ohe_album_type_col].value_counts(dropna=False)}")
        df = pd.get_dummies(df, columns=[ohe_album_type_col], prefix='dART', dummy_na=False)
        logging.info(f"One-hot encoded '{ohe_album_type_col}'. New columns: {[col for col in df.columns if col.startswith('dART_')]}")
    else:
        logging.warning(f"'{ohe_album_type_col}' column not found for one-hot encoding.")
        
    df.sort_values("date", inplace=True)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning rules: drops irrelevant/zero-variance columns, handles NaNs."""
    drop_extra = [
        "mbid", "mb_recording_id", "error", 
        "acoustic_ab_genre_tzanetakis_value", # Example specific AB value column
        "deezer_track_id", "deezer_album_id", "deezer_genres" 
    ]
    to_drop_extra_present = [c for c in drop_extra if c in df.columns]
    if to_drop_extra_present:
        df.drop(columns=to_drop_extra_present, inplace=True)
        logging.info(f"Dropped irrelevant/ID cols: {to_drop_extra_present}")

    drop_zero_variance = [ # These are often low variance or redundant after other processing
        "acoustic_ab_length", "acoustic_ab_tempo_confidence",
        "acoustic_ab_tonal_confidence", "acoustic_ab_rhythm_level",
        "acoustic_ab_bpm", "acoustic_ab_mood_aggressive_prob"            
    ]
    to_drop_zero_present = [c for c in drop_zero_variance if c in df.columns]
    if to_drop_zero_present:
        df.drop(columns=to_drop_zero_present, inplace=True)
        logging.info(f"Dropped specified potentially zero-variance/uninformative AcousticBrainz cols: {to_drop_zero_present}")

    dz_num_cols = [
        c for c in df.columns
        if c.startswith("deezer_") and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("deezer_genre_") 
        and c != 'deezer_release_date' 
    ]
    if dz_num_cols:
        df[dz_num_cols] = df[dz_num_cols].replace(0, np.nan)
        logging.info(f"Replaced zeros with NaN in {len(dz_num_cols)} numeric Deezer columns (excluding genres and release_date).")

    if "deezer_genre_list" in df.columns: 
        df.drop(columns=["deezer_genre_list"], inplace=True, errors='ignore')

    for col in ["album_duration", "album_track_count", "deezer_album_duration", "deezer_album_track_count", "lastfm_duration_ms"]: 
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]): 
            # For these specific count/duration columns, 0 might be a valid value or an indicator of missing/irrelevant.
            # If 0 should be NaN for these as well:
            df[col] = df[col].replace(0, np.nan)
            logging.info(f"Replaced 0 with NaN in '{col}'.")
            pass 

    # --- Handling AcousticBrainz _prob and _value columns ---
    prob_cols = [c for c in df.columns if c.endswith("_prob") and c.startswith("acoustic_")]
    if prob_cols:
        logging.info(f"Processing {len(prob_cols)} AcousticBrainz _prob columns...")
        # Calculate unique ratio to identify low variance probability columns
        # Ensure there are rows to calculate nunique and len, and prob_cols are actually in df
        valid_prob_cols_for_ratio = [col for col in prob_cols if col in df.columns]
        if not df.empty and valid_prob_cols_for_ratio:
            uniq_ratio = df[valid_prob_cols_for_ratio].nunique(dropna=True).astype(float) / len(df)
            drop_low_variance_probs = uniq_ratio[uniq_ratio < 0.01].index.tolist() # Threshold for low variance
        else:
            drop_low_variance_probs = []

        if drop_low_variance_probs:
            df.drop(columns=drop_low_variance_probs, inplace=True, errors='ignore')
            logging.info(f"Dropped low variance AcousticBrainz _prob columns: {drop_low_variance_probs}")
        
        # Process remaining (kept) probability columns
        kept_prob_cols = [c for c in prob_cols if c not in drop_low_variance_probs and c in df.columns]
        if kept_prob_cols:
            for p_col in kept_prob_cols:
                # Binarize: > 0.5 becomes 1, else 0. Handle NaNs by keeping them NaN.
                df[p_col] = np.where(df[p_col] > 0.5, 1, np.where(df[p_col].isna(), np.nan, 0))
                logging.info(f"Binarized AcousticBrainz _prob column '{p_col}'.")

                # Drop corresponding '_value' column (this is the "drop the flags" part for these pairs)
                val_col = p_col.replace("_prob", "_value")
                if val_col in df.columns:
                    df.drop(columns=[val_col], inplace=True, errors='ignore')
                    logging.info(f"Dropped corresponding AcousticBrainz _value column '{val_col}'.")
            logging.info(f"Finished processing and binarizing kept AcousticBrainz _prob columns.")
        else:
            logging.info("No AcousticBrainz _prob columns kept after low variance filter or none found initially.")

    core_identifiers = ["artist", "song", "date"] 
    if "mbid" in df.columns: core_identifiers.append("mbid")
    
    feature_columns = [c for c in df.columns if c not in core_identifiers 
                       and not c.startswith("dART_") 
                       and not c.startswith("deezer_genre_")
                       and not c.startswith("lfm_tag_")] 
    
    if feature_columns: 
        empty_feature_rows = df[feature_columns].isna().all(axis=1).sum()
        logging.info(f"{empty_feature_rows:,} rows have all NaN in their feature columns (excluding identifiers and OHE groups).")

    if "spotify_streams" in df.columns:
        before_filter_streams = len(df)
        df = df[
            df["spotify_streams"].notna() & (df["spotify_streams"] > 0) 
        ].copy() 
        logging.info(
            f"Dropped {before_filter_streams - len(df):,} rows with NaN or non-positive 'spotify_streams'."
        )
    else:
        logging.warning("'spotify_streams' column not found; cannot filter based on it.")

    return df

def load(df: pd.DataFrame, processed_dir: Path) -> None:
    """Saves the final modeling dataset."""
    out_path = processed_dir / "model_dataset_weekly.csv"
    df.to_csv(out_path, index=False) 
    
    rel_path = "UNKNOWN_PATH"
    try:
        rel_path = out_path.relative_to(root)
    except ValueError:
        rel_path = out_path
    print(f"✅ Modeling dataset saved ➔ {rel_path} ({len(df):,} rows, {df.shape[1]} cols)")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge, enrich, clean, and save weekly modeling dataset"
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=config.PROCESSED_DIR, 
        help="Directory containing processed CSVs (master_feature_table.csv, merged_weekly.csv)",
    )
    return p.parse_args()

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    args = parse_args()
    feats_df, weekly_df = extract(args.processed_dir)
    transformed_df = transform(feats_df, weekly_df)
    cleaned_df = clean(transformed_df)
    load(cleaned_df, args.processed_dir)

if __name__ == "__main__":
    main()
