# src/etl/forecasting/05_aggregate_genre_data.py

import pandas as pd
from pathlib import Path
import sys
import numpy as np

# --- Path Setup ---
try:
    project_root = Path(__file__).resolve().parents[3]
    src_root = Path(__file__).resolve().parents[2]
except IndexError:
    print("Error: Could not determine project_root or src_root.")
    project_root = Path.cwd(); src_root = Path.cwd() / "src"
    print(f"Attempting to use CWD as project_root: {project_root}")

if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_root) not in sys.path: sys.path.insert(0, str(src_root))
# --- End Path Setup ---

try:
    from etl.forecasting.common.load_data import load_processed_dataset # To get the full dataset
    import config # For output directory
except ModuleNotFoundError as e:
    print(f"Failed to import common modules or config: {e}"); raise

TARGET_COLUMN = 'spotify_streams'
# Define where to save the aggregated genre data
AGGREGATED_GENRE_DATA_DIR = config.PROCESSED_DIR / "genre_aggregated_timeseries"
AGGREGATED_GENRE_DATA_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    print("--- Running 05_aggregate_genre_data.py ---")
    full_df = load_processed_dataset() # Loads model_dataset_weekly.csv

    if full_df.empty:
        print("Loaded dataset (model_dataset_weekly.csv) is empty. Exiting.")
        sys.exit()
    
    if 'date' not in full_df.columns or TARGET_COLUMN not in full_df.columns:
        print(f"Required columns 'date' or '{TARGET_COLUMN}' not in DataFrame. Exiting.")
        sys.exit()

    # Convert date column to datetime if it's not already (load_processed_dataset should do this)
    full_df['date'] = pd.to_datetime(full_df['date'])

    # Identify one-hot encoded Deezer genre columns
    all_cols = full_df.columns.tolist()
    identified_genre_cols = []
    potential_genre_cols = [col for col in all_cols if col.startswith('deezer_genre_')]

    if not potential_genre_cols:
        print("Warning: No columns found starting with 'deezer_genre_'. Cannot identify genre columns for aggregation.")
    else:
        for col_name in potential_genre_cols:
            if pd.api.types.is_numeric_dtype(full_df[col_name].dtype):
                unique_vals = full_df[col_name].dropna().unique()
                is_binary_like = True 
                if not unique_vals.size > 0 : is_binary_like = False
                for val in unique_vals:
                    if val not in [0, 1, 0.0, 1.0]: 
                        is_binary_like = False; break
                if is_binary_like: identified_genre_cols.append(col_name)
            elif full_df[col_name].dtype == 'bool':
                 unique_vals_bool = full_df[col_name].dropna().unique()
                 if all(val in [True, False] for val in unique_vals_bool):
                     identified_genre_cols.append(col_name)
    
    if not identified_genre_cols:
        print("Critical: Failed to identify any one-hot encoded 'deezer_genre_*' columns. Cannot proceed with genre aggregation. Exiting.")
        sys.exit()
    else:
        print(f"Identified {len(identified_genre_cols)} one-hot encoded Deezer genre columns for aggregation.")
        print(f"Sample genre columns: {identified_genre_cols[:min(5, len(identified_genre_cols))]}")

    aggregated_genre_dfs = {}

    for genre_col_name in identified_genre_cols:
        clean_genre_name = genre_col_name.replace('deezer_genre_', '').replace('Ã±', 'n') # Clean name for filename
        
        # Filter songs belonging to this genre
        # A song has this genre if the respective one-hot encoded column is 1 (or True)
        genre_songs_df = full_df[full_df[genre_col_name] == 1]
        
        if genre_songs_df.empty:
            print(f"No songs found for genre: {clean_genre_name} (column: {genre_col_name}). Skipping.")
            continue
            
        # Aggregate streams by date for this genre
        # Ensure we sum up streams from *different songs* within the same genre for the same date
        genre_aggregated_series = genre_songs_df.groupby('date')[TARGET_COLUMN].sum().reset_index()
        genre_aggregated_series.rename(columns={TARGET_COLUMN: f'{clean_genre_name}_total_streams'}, inplace=True)
        
        # Ensure dates are unique and sorted
        genre_aggregated_series = genre_aggregated_series.sort_values(by='date').set_index('date')
        
        if genre_aggregated_series.empty:
            print(f"Aggregated series for {clean_genre_name} is empty. Skipping.")
            continue

        # Save each genre's aggregated time series to a separate CSV file
        output_file_path = AGGREGATED_GENRE_DATA_DIR / f"genre_ts_{clean_genre_name}.csv"
        genre_aggregated_series.to_csv(output_file_path) # Saves with 'date' as index
        print(f"Saved aggregated time series for genre '{clean_genre_name}' to {output_file_path} ({len(genre_aggregated_series)} weeks)")
        aggregated_genre_dfs[clean_genre_name] = genre_aggregated_series

    if not aggregated_genre_dfs:
        print("No genre time series were successfully aggregated and saved.")
    else:
        print(f"\nSuccessfully aggregated and saved time series for {len(aggregated_genre_dfs)} genres.")
        print(f"Data saved in: {AGGREGATED_GENRE_DATA_DIR}")

    print("\n--- 05_aggregate_genre_data.py finished ---")