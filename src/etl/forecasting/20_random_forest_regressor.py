# src/etl/forecasting/20_random_forest_regressor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from sklearn.ensemble import RandomForestRegressor
# Import tools for robust tuning and validation
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit 
from scipy.stats import randint 
from tqdm import tqdm

# --- Path Setup & Imports ---
try:
    project_root = Path(__file__).resolve().parents[3]
    src_root = Path(__file__).resolve().parents[2]
except IndexError:
    print("Error: Could not determine project_root or src_root.")
    project_root = Path.cwd(); src_root = Path.cwd() / "src"
    print(f"Attempting to use CWD as project_root: {project_root}")
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_root) not in sys.path: sys.path.insert(0, str(src_root))
try:
    from etl.forecasting.common.load_data import load_processed_dataset
    from etl.forecasting.common.evaluation import evaluate_forecast
    import config 
except ModuleNotFoundError as e: print(f"Failed to import common modules: {e}"); raise
# --- End Path Setup & Imports ---

TARGET_COLUMN = 'spotify_streams'
# --- Output Directories ---
FORECAST_CSV_DIR = project_root / "outputs" / "forecasts"; FORECAST_CSV_DIR.mkdir(parents=True, exist_ok=True) 
FORECAST_PLOT_DIR = project_root / "outputs" / "plots"; FORECAST_PLOT_DIR.mkdir(parents=True, exist_ok=True) 
# --- End Output Directories ---

# --- Configuration for FULL RUN with ROBUST TUNING ---
TEST_SET_FRACTION = 0.15      # Fraction of data for final hold-out test set
MAX_SONGS_TO_PROCESS = 200   
# --- Tuning Configuration ---
N_ITER_RANDOM_SEARCH = 10    
N_CV_SPLITS = 3            
# --- End Configuration ---

warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

# (prepare_ml_data function remains the same - using the correct drop_cols based on your CSV)
def prepare_ml_data(df, target_col):
    """ Prepares features (X) and target (y) """
    drop_cols = [
        'artist', 'song', 'date', 'deezer_release_date','spotify_position', 'spotify_peakposition', 
        'spotify_weeksonchart', 'spotify_days_on_chart', 'apple_pos', 'apple_days', 'apple_pk',
        'avg_position', 'avg_peak_position', 'combined_days_on_chart','deezer_album_record_type',
        'spotify_lastweek' # Keeping this lagged feature
    ]
    # Ensure target exists before trying to drop
    if target_col not in df.columns:
        print(f"  Error: Target column '{target_col}' not found in DataFrame.")
        return pd.DataFrame(), pd.Series()

    y = df[target_col].copy(); 
    X_candidate = df.drop(columns=[target_col] + [col for col in drop_cols if col in df.columns], errors='ignore')
    X_numeric = X_candidate.select_dtypes(include=np.number)
    if X_numeric.shape[1] == 0: 
        print("  Warning: No numeric features selected after dropping columns.")
        return pd.DataFrame(), pd.Series() 
    X_imputed = X_numeric.fillna(0); 
    non_numeric_cols_remaining = X_imputed.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols_remaining) > 0: 
        print(f"  Warning: Dropping non-numeric columns found after selection: {non_numeric_cols_remaining}.")
        X_imputed = X_imputed.drop(columns=non_numeric_cols_remaining)
    return X_imputed, y

if __name__ == '__main__':
    print("--- Running 20_random_forest_regressor.py (Robust Tuning with TimeSeriesSplit) ---")
    print(f"Config: Max Songs={MAX_SONGS_TO_PROCESS}, Tuning Iterations={N_ITER_RANDOM_SEARCH}, CV Splits={N_CV_SPLITS}")
    full_df = load_processed_dataset()
    if full_df.empty: print("Loaded dataset empty."); sys.exit()

    # (Genre Column Identification logic remains the same)
    all_cols = full_df.columns.tolist(); identified_genre_cols = []
    potential_genre_cols = [col for col in all_cols if col.startswith('deezer_genre_')]
    if potential_genre_cols: # ... (rest of genre identification logic same as before) ...
         for col_name in potential_genre_cols:
            if pd.api.types.is_numeric_dtype(full_df[col_name].dtype):
                unique_vals = full_df[col_name].dropna().unique()
                is_binary_like = True; 
                if not unique_vals.size > 0 : is_binary_like = False
                for val in unique_vals: 
                    if val not in [0, 1, 0.0, 1.0]: is_binary_like = False; break
                if is_binary_like: identified_genre_cols.append(col_name)
            elif full_df[col_name].dtype == 'bool':
                 if all(val in [True, False] for val in full_df[col_name].dropna().unique()): identified_genre_cols.append(col_name)
    if not identified_genre_cols: print("Warning: No one-hot encoded genre columns identified.")
    else: print(f"Identified {len(identified_genre_cols)} Deezer genre columns.")

    all_song_predictions_info = [] 
    song_level_results_rf = []
    best_params_log = [] 
    
    # (Selecting songs to process remains the same)
    unique_songs_tuples = full_df[['artist', 'song']].drop_duplicates().values.tolist()
    songs_to_process_list = []
    if MAX_SONGS_TO_PROCESS is not None and MAX_SONGS_TO_PROCESS < len(unique_songs_tuples):
        print(f"Processing a subset of {MAX_SONGS_TO_PROCESS} songs.")
        song_counts = full_df.groupby(['artist', 'song']).size().sort_values(ascending=False)
        songs_to_process_list = list(song_counts.head(MAX_SONGS_TO_PROCESS).index)
    else: songs_to_process_list = [(r[0], r[1]) for r in unique_songs_tuples]
    print(f"Total songs to process: {len(songs_to_process_list)}")
    if not songs_to_process_list: sys.exit("No songs found to process.")

    # --- Define Hyperparameter Search Space for RF ---
    param_dist_rf = {
        'n_estimators': randint(50, 150), 'max_depth': randint(5, 20),
        'min_samples_split': randint(5, 20), 'min_samples_leaf': randint(3, 15),
    }
    # --- End Search Space Definition ---

    # --- Main Loop: Process Selected Songs ---
    model_name_agg = "RF_tscv_tuned" 
    for i, (artist_name, song_title) in enumerate(tqdm(songs_to_process_list, desc=f"Processing Songs ({model_name_agg})", unit="song")):
        
        song_df_original = full_df[(full_df['artist'] == artist_name) & (full_df['song'] == song_title)]
        # Need enough data for train/test split AND TimeSeriesSplit internal splits
        min_required_data = N_CV_SPLITS + 10 # Heuristic minimum
        if song_df_original.empty or len(song_df_original) < min_required_data: continue 

        X_song, y_song = prepare_ml_data(song_df_original.copy(), TARGET_COLUMN) 
        if X_song.empty or X_song.shape[1] == 0 or y_song.empty or y_song.isna().all(): continue 
        
        # --- Fixed-period split for consistency ---
        N_TEST_PERIODS = 8  # Matches genre-level naive model
        if len(X_song) <= N_TEST_PERIODS:
            tqdm.write(f"DEBUG: Skipping - Not enough data for {N_TEST_PERIODS} test periods.")
            continue

        n_train_points = len(X_song) - N_TEST_PERIODS
        n_test_points = N_TEST_PERIODS

        
        # Check if enough training points for TimeSeriesSplit
        if n_test_points < 1 or n_train_points < N_CV_SPLITS + 2: continue 

        X_train_full, X_test = X_song.iloc[:n_train_points], X_song.iloc[n_train_points:]
        y_train_full, y_test = y_song.iloc[:n_train_points], y_song.iloc[n_train_points:]
        
        # Capture info needed for genre aggregation later
        test_info_capture_df = song_df_original.iloc[n_train_points:].copy() 
        test_info_df_for_genre_agg = pd.DataFrame(index=test_info_capture_df.index); test_info_df_for_genre_agg['date'] = test_info_capture_df['date']
        for gc_col in identified_genre_cols: 
            if gc_col in test_info_capture_df.columns: test_info_df_for_genre_agg[gc_col] = test_info_capture_df[gc_col]
        test_info_df_for_genre_agg['actual_streams'] = y_test.values; test_info_df_for_genre_agg['artist'] = artist_name; test_info_df_for_genre_agg['song'] = song_title   

        if X_train_full.empty or y_train_full.empty or X_test.empty or X_train_full.shape[1] == 0: continue 
        
        # --- Robust Hyperparameter Tuning using TimeSeriesSplit ---
        best_rf_model = None
        tuning_successful = False
        best_params_found = 'default_fallback' # Start with fallback assumption
        cv_score_found = np.nan

        try:
            # Define the time series cross-validator
            tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS) 
            
            # Base RF estimator
            rf = RandomForestRegressor(random_state=42, n_jobs=1) 

            # Randomized search setup
            random_search = RandomizedSearchCV(
                estimator=rf, 
                param_distributions=param_dist_rf, 
                n_iter=N_ITER_RANDOM_SEARCH, #
                cv=tscv, 
                scoring='neg_mean_absolute_error', 
                n_jobs=1, 
                random_state=42,
                verbose=0,
                error_score=np.nan 
            )
                
            # Fit the random search on the full training data
            random_search.fit(X_train_full, y_train_full.fillna(0)) 
                
            # Check if the best score found is valid (not NaN)
            if not pd.isna(random_search.best_score_):
                 best_rf_model = random_search.best_estimator_
                 # Re-fit the best estimator on the *entire* training set
                 best_rf_model.fit(X_train_full, y_train_full.fillna(0))
                 tuning_successful = True
                 best_params_found = random_search.best_params_
                 cv_score_found = random_search.best_score_ # This is the negative MAE
                 tqdm.write(f"  Tuning success: {artist_name[:15]}... Score={cv_score_found:.0f}") # Less verbose success
            else:
                tqdm.write(f"  Tuning completed with NaN score for {artist_name[:15]}... Using default.")
                # Fallback handled below if tuning_successful is False

        except ValueError as ve:
             # Catch specific TimeSeriesSplit errors if train set is too small after split
             tqdm.write(f"  CV Split Error during tuning for {artist_name[:15]}... ({ve}). Using default.")
        except Exception as e:
            tqdm.write(f"  RF Tuning/Fit failed unexpectedly: {e}. Using default params.")
            
        # If tuning failed or gave NaN score, use default params
        if not tuning_successful:
            best_params_found = 'default_fallback' # Log fallback reason
            cv_score_found = np.nan
            best_rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=10, min_samples_leaf=5)
            try:
                best_rf_model.fit(X_train_full, y_train_full.fillna(0))
            except Exception as fit_e:
                 tqdm.write(f"  Default RF fit failed: {fit_e}. Skipping song.")
                 continue # Skip if even default fails
        
        best_params_log.append({'artist': artist_name, 'song': song_title, 'best_params': best_params_found, 'cv_score': cv_score_found})
        # --- End Tuning ---

        # --- Predict, Evaluate, Plot, Store ---
        try:
            predictions_rf = best_rf_model.predict(X_test)
            pred_col_name = f'predicted_streams_{model_name_agg}'
            test_info_df_for_genre_agg[pred_col_name] = predictions_rf 
            all_song_predictions_info.append(test_info_df_for_genre_agg)
        except Exception as pred_e:
            tqdm.write(f"  Prediction failed for {artist_name[:15]}...: {pred_e}")
            continue 
        valid_test_indices = y_test.notna()
        if not valid_test_indices.any(): continue 
        song_eval_metrics = evaluate_forecast(y_test[valid_test_indices], predictions_rf[valid_test_indices], model_name=f"RF ({artist_name[:15]}...)", print_results=False)
        current_song_eval = {'artist': artist_name, 'song': song_title, **song_eval_metrics}; del current_song_eval['model']
        song_level_results_rf.append(current_song_eval)
        
        # --- Plotting (Only first successful song) ---
        if len(song_level_results_rf) == 1: 
             tqdm.write(f"  Generating plot for first successful song: {artist_name} - {song_title}")
             try: 
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(X_train_full.index, y_train_full, label='Train', alpha=0.7) 
                ax.plot(X_test.index, y_test, label='Test Actual', marker='.')
                ax.plot(X_test.index, predictions_rf, label=f'{model_name_agg} Forecast', linestyle='--')
                ax.set_title(f"{model_name_agg}: {artist_name} - {song_title}", fontsize=14)
                ax.legend(); ax.grid(True); plt.xticks(rotation=45); plt.tight_layout()
                plot_filename = FORECAST_PLOT_DIR / f"forecast_plot_{model_name_agg}_{artist_name}_{song_title}.png"
                plt.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig) 
             except Exception as plot_err: 
                 tqdm.write(f"    Warning: Plot failed: {plot_err}")
                 if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        # --- End Plotting ---
            
    # --- Genre Aggregation / Saving ---
    if all_song_predictions_info:
        print(f"\n--- Aggregating {model_name_agg} Predictions ---")
        all_predictions_df_combined = pd.concat(all_song_predictions_info, ignore_index=True)
        genre_level_results = []
        if not identified_genre_cols: print("No genre columns identified.")
        else:
            for genre_col_name in identified_genre_cols:
                if genre_col_name not in all_predictions_df_combined.columns: continue
                genre_specific_predictions_df = all_predictions_df_combined[all_predictions_df_combined[genre_col_name] == 1].copy() 
                if genre_specific_predictions_df.empty: continue
                pred_col_name = f'predicted_streams_{model_name_agg}'
                if pred_col_name not in genre_specific_predictions_df.columns: continue
                genre_specific_predictions_df['predicted_streams_filled'] = genre_specific_predictions_df[pred_col_name].fillna(0)
                genre_agg_actuals = genre_specific_predictions_df.groupby('date')['actual_streams'].sum()
                genre_agg_predictions = genre_specific_predictions_df.groupby('date')['predicted_streams_filled'].sum()
                aligned_genre_df = pd.DataFrame({'actual': genre_agg_actuals, 'predicted': genre_agg_predictions}).dropna()
                if aligned_genre_df.empty or len(aligned_genre_df) < 1: continue
                clean_genre_name = genre_col_name.replace('deezer_genre_', '')
                genre_eval = evaluate_forecast(aligned_genre_df['actual'], aligned_genre_df['predicted'], model_name=f"{model_name_agg} ({clean_genre_name})", print_results=False)
                genre_level_results.append({'genre': clean_genre_name, **genre_eval}); del genre_level_results[-1]['model']
        if genre_level_results:
            genre_summary_final_df = pd.DataFrame(genre_level_results)
            print(f"\n--- {model_name_agg} - Genre-Level Forecast Results Summary ---")
            print(genre_summary_final_df.sort_values(by='mape'))
            metrics_filename_genre = FORECAST_CSV_DIR / f"genre_{model_name_agg}_metrics_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
            genre_summary_final_df.to_csv(metrics_filename_genre, index=False)
            print(f"\nGenre metrics saved to: {metrics_filename_genre.relative_to(project_root)}")
            avg_genre_mape = genre_summary_final_df['mape'].mean(skipna=True)
            if not pd.isna(avg_genre_mape): print(f"Overall Average Genre MAPE ({model_name_agg}): {avg_genre_mape:.2f}%")
            genre_predictions_to_save = []
            for genre_col_name in identified_genre_cols:
                 if genre_col_name not in all_predictions_df_combined.columns: continue
                 genre_df_temp = all_predictions_df_combined[all_predictions_df_combined[genre_col_name] == 1].copy()
                 if not genre_df_temp.empty:
                     pred_col = f'predicted_streams_{model_name_agg}'
                     if pred_col not in genre_df_temp.columns: continue
                     genre_pred_agg = genre_df_temp.groupby('date').agg(actual_streams_total=('actual_streams', 'sum'), predicted_streams_total=(pred_col, 'sum')).reset_index()
                     genre_pred_agg['genre'] = genre_col_name.replace('deezer_genre_','')
                     genre_predictions_to_save.append(genre_pred_agg[['date','genre','actual_streams_total','predicted_streams_total']])
            if genre_predictions_to_save:
                 combined_genre_predictions_df = pd.concat(genre_predictions_to_save, ignore_index=True)
                 predictions_filename_genre = FORECAST_CSV_DIR / f"genre_{model_name_agg}_predictions_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
                 combined_genre_predictions_df.to_csv(predictions_filename_genre, index=False)
                 print(f"Aggregated genre predictions saved to: {predictions_filename_genre.relative_to(project_root)}")

        elif identified_genre_cols: print("No genre-level results to display.")
    else: print("\nNo song predictions made.")

    # --- Saving Song Level Results ---
    # if song_level_results_rf:
    #     song_summary_final_df = pd.DataFrame(song_level_results_rf)
    #     print(f"\n--- {model_name_agg} - Song-Level Forecast Results Summary (Sample) ---")
    #     print(song_summary_final_df.head()) 
    #     metrics_filename_song = FORECAST_CSV_DIR / f"song_{model_name_agg}_metrics_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
    #     # song_summary_final_df.to_csv(metrics_filename_song, index=False)
    #     print(f"\nSong metrics saved to: {metrics_filename_song.relative_to(project_root)}")
    #     avg_song_mape = song_summary_final_df['mape'].mean(skipna=True)
    #     if not pd.isna(avg_song_mape): print(f"Average Song-Level MAPE ({model_name_agg}): {avg_song_mape:.2f}%")
        
    #     if best_params_log:
    #         params_df = pd.DataFrame(best_params_log)
    #         params_filename = FORECAST_CSV_DIR / f"song_{model_name_agg}_best_params_log_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
    #         params_df.to_csv(params_filename, index=False)
    #         print(f"Best params log saved to: {params_filename.relative_to(project_root)}")
            
    #     # Save detailed song predictions
    #     if all_song_predictions_info:
    #          all_preds_df = pd.concat(all_song_predictions_info, ignore_index=True)
    #          preds_filename_song = FORECAST_CSV_DIR / f"song_{model_name_agg}_predictions_songs{len(songs_to_process_list)}_test{int(TEST_SET_FRACTION * 100)}.csv" # Use fraction in name maybe
    #          cols_to_save = ['date', 'artist', 'song', 'actual_streams', f'predicted_streams_{model_name_agg}'] + identified_genre_cols
    #          all_preds_df[[col for col in cols_to_save if col in all_preds_df.columns]].to_csv(preds_filename_song, index=False)
    #          print(f"Detailed song predictions saved to: {preds_filename_song.relative_to(project_root)}")

    print(f"\n--- 20_random_forest_regressor.py ({model_name_agg}) finished ---")