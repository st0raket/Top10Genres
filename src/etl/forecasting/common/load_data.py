# src/etl/forecasting/common/load_data.py

import pandas as pd
from pathlib import Path
import sys
import numpy as np

# --- Path Setup ---
# This script is in: Capstone/src/etl/forecasting/common/load_data.py
# Project root (where config.py is) is 4 levels up.
# The 'src' directory (for utils, etl modules) is 3 levels up.

try:
    # Define project_root based on this file's location
    # common (0) -> forecasting (1) -> etl (2) -> src (3) -> Capstone (4)
    project_root = Path(__file__).resolve().parents[4]
except IndexError:
    print("Error: Could not determine project_root. Relative pathing might be incorrect.")
    print("Ensure this script is correctly placed within the project structure.")
    # Fallback or raise error, for now, let's try CWD as a last resort, though less ideal
    project_root = Path.cwd()
    print(f"Attempting to use CWD as project_root: {project_root}")


# Add project_root to sys.path for importing 'config'
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src_path to sys.path for importing from 'utils' or other 'src' submodules
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
# --- End Path Setup ---


# Now, import config should work
try:
    import config
    # If you had utils/features.py, you could also do:
    # from utils.features import some_utility_function
except ModuleNotFoundError as e:
    print(f"FATAL: Could not import 'config' or other project modules. Error: {e}")
    print(f"PROJECT_ROOT was set to '{project_root}'.")
    print(f"SRC_PATH was set to '{src_path}'.")
    print(f"Ensure 'config.py' exists in '{project_root}' and other modules are in '{src_path}'.")
    print(f"Current sys.path: {sys.path}")
    raise

PROCESSED_DIR = config.PROCESSED_DIR
MODEL_DATASET_PATH = PROCESSED_DIR / "model_dataset_weekly.csv"

def load_processed_dataset():
    """Loads the model_dataset_weekly.csv file."""
    print(f"Attempting to load dataset from: {MODEL_DATASET_PATH}")
    if not MODEL_DATASET_PATH.exists():
        print(f"Dataset not found at {MODEL_DATASET_PATH}.")
        print(f"PROCESSED_DIR ({PROCESSED_DIR}) exists: {PROCESSED_DIR.exists()}")
        
        # Create a dummy file for testing if it doesn't exist.
        # REMOVE OR COMMENT THIS OUT ONCE YOUR ETL PIPELINE IS WORKING.
        print("Creating a dummy model_dataset_weekly.csv for testing load_data.py.")
        print("Please run your full ETL pipeline (60_merge_timeseries.py) to generate the actual dataset.")
        
        if not PROCESSED_DIR.exists():
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {PROCESSED_DIR}")

        dummy_data = {
            'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-01', '2023-01-08', '2023-01-22']),
            'artist': ['Artist A', 'Artist A', 'Artist A', 'Artist B', 'Artist B', 'Artist A'],
            'song': ['Song X', 'Song X', 'Song X', 'Song Y', 'Song Y', 'Song Z'],
            'spotify_streams': [100, 110, 120, 200, 210, 50],
            'feature1': [1, 2, 3, 4, 5, 6],
            'lag_1_spotify_streams': [np.nan, 100, 110, np.nan, 200, np.nan]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(MODEL_DATASET_PATH, index=False)
        print(f"Dummy dataset created at {MODEL_DATASET_PATH}")
    
    try:
        df = pd.read_csv(MODEL_DATASET_PATH)
    except Exception as e:
        print(f"Error reading CSV {MODEL_DATASET_PATH}: {e}")
        raise

    if 'date' not in df.columns:
        print("Warning: 'date' column not found in the dataset. Cannot convert to datetime or sort by date.")
    else:
        df['date'] = pd.to_datetime(df['date'])

    required_sort_cols = ['artist', 'song', 'date']
    if all(col in df.columns for col in required_sort_cols):
        df.sort_values(by=['artist', 'song', 'date'], inplace=True)
    elif 'date' in df.columns:
        df.sort_values(by=['date'], inplace=True)
    else:
        print("Warning: Could not sort DataFrame as key date/identifier columns are missing.")
    return df

def get_song_timeseries(df, artist_name, song_name, target_column='spotify_streams', date_column='date'):
    """Extracts time series for a specific song."""
    if not all(col in df.columns for col in ['artist', 'song']):
        print(f"DataFrame is missing 'artist' or 'song' columns. Cannot filter for {artist_name} - {song_name}.")
        return pd.DataFrame()
    
    song_df_filtered = df[(df['artist'] == artist_name) & (df['song'] == song_name)].copy()

    if song_df_filtered.empty:
        print(f"No data found for artist '{artist_name}' and song '{song_name}'.")
        return pd.DataFrame()

    if date_column not in song_df_filtered.columns:
        print(f"Date column '{date_column}' not found in filtered data for {artist_name} - {song_name}.")
        return pd.DataFrame()
    
    song_df_filtered.set_index(date_column, inplace=True)

    if target_column not in song_df_filtered.columns:
        print(f"Target column '{target_column}' not found in data for {artist_name} - {song_name} after setting index.")
        return pd.DataFrame() 

    return song_df_filtered[[target_column]]

if __name__ == '__main__':
    print(f"--- Running load_data.py ---")
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Determined project_root: {project_root}")
    print(f"Determined src_path: {src_path}")
    print(f"PYTHONPATH includes project_root: {str(project_root) in sys.path}")
    print(f"PYTHONPATH includes src_path: {str(src_path) in sys.path}")
    print(f"Attempting to import 'config' from {project_root / 'config.py'}")

    full_df = load_processed_dataset()
    print("\n--- Dataset Loading Summary ---")
    if not full_df.empty:
        print(full_df.info())
        # print(f"\nFirst 5 rows of loaded data:\n{full_df.head()}") # Can be verbose

        if 'artist' in full_df.columns and 'song' in full_df.columns and len(full_df) > 0:
            song_counts = full_df.groupby(['artist', 'song']).size()
            if not song_counts.empty:
                # Try to pick an example song that has more than 1 entry if possible
                sorted_song_counts = song_counts.sort_values(ascending=False)
                if not sorted_song_counts.empty:
                    example_artist, example_song = sorted_song_counts.index[0]
                    print(f"\n--- Example Song Time Series ({example_artist} - {example_song}) ---")
                    song_ts = get_song_timeseries(full_df, example_artist, example_song)
                    if not song_ts.empty:
                        print(song_ts.head())
                    else:
                        print(f"Could not retrieve time series for example song: {example_artist} - {example_song}")
            else:
                print("No song groups found to pick an example.")
        else:
            print("\nCould not select an example song: 'artist' or 'song' column missing, or DataFrame empty.")
    else:
        print("Loaded DataFrame is empty. Further example operations skipped.")
    print("--- load_data.py finished ---")