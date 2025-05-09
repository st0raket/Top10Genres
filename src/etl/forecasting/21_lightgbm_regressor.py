# src/etl/forecasting/21_lightgbm_regressor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
from tqdm import tqdm

try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False
    print("CRITICAL: LightGBM not installed. This script requires LightGBM.")

# --- Path Setup & Imports ---
try:
    project_root = Path(__file__).resolve().parents[3]
    src_root = Path(__file__).resolve().parents[2]
except IndexError:
    print("Error: Could not determine project_root or src_root from __file__.")
    project_root = Path.cwd()
    if project_root.name == "forecasting" and project_root.parent.name == "etl" and project_root.parent.parent.name == "src":
        project_root = project_root.parents[3]
    elif project_root.name == "etl" and project_root.parent.name == "src":
        project_root = project_root.parents[2]
    elif project_root.name == "src":
        project_root = project_root.parents[1]
    src_root = project_root / "src"
    print(f"Attempting to use CWD-derived project_root: {project_root}")
    print(f"Attempting to use CWD-derived src_root: {src_root}")

if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_root) not in sys.path: sys.path.insert(0, str(src_root))

try:
    from etl.forecasting.common.load_data import load_processed_dataset
    from etl.forecasting.common.evaluation import evaluate_forecast
    import config
except ModuleNotFoundError as e:
    print(f"Failed to import common modules or config from configured sys.path: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Please ensure 'config.py' is in the project_root and common modules are in 'src/etl/forecasting/common/'")
    raise
# --- End Path Setup & Imports ---

TARGET_COLUMN = 'spotify_streams'
# --- Output Directories ---
FORECAST_CSV_DIR = project_root / "outputs" / "forecasts"
FORECAST_PLOT_DIR = project_root / "outputs" / "plots" /"model"
FORECAST_CSV_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_PLOT_DIR.mkdir(parents=True, exist_ok=True)
# --- End Output Directories ---

# --- Configuration ---
TEST_SET_FRACTION = 0.15
MAX_SONGS_TO_PROCESS = None # ADJUST FOR SPEED
# --- Tuning Configuration ---
N_ITER_RANDOM_SEARCH = 10
N_CV_SPLITS_LGBM_TUNING = 3
# --- End Configuration ---

# Filter general warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if LGBM_INSTALLED:
    try:
        warnings.filterwarnings("ignore", category=lgb.LightGBMWarning)
    except AttributeError:
        print("Note: lgb.LightGBMWarning not found, specific LightGBM warnings might not be suppressed.")


def prepare_ml_data(df, target_col):
    """ Prepares features (X) and target (y) """
    drop_cols = [
        'artist', 'song', 'date', 'deezer_release_date','spotify_position', 'spotify_peakposition',
        'spotify_weeksonchart', 'spotify_days_on_chart', 'apple_pos', 'apple_days', 'apple_pk',
        'avg_position', 'avg_peak_position', 'combined_days_on_chart','deezer_album_record_type', # Original, pre-OHE
        'spotify_lastweek' # This might be spotify_streams_lag_1
    ]
    if target_col not in df.columns:
        print(f"  Error: Target column '{target_col}' not found in DataFrame.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    y = df[target_col].copy()
    X_candidate = df.drop(columns=[target_col] + [col for col in drop_cols if col in df.columns], errors='ignore')
    X_numeric = X_candidate.select_dtypes(include=np.number)

    if X_numeric.shape[1] == 0:
        print("  Warning: No numeric features selected after dropping columns and selecting numeric types.")
        return pd.DataFrame(), pd.Series(dtype='float64')

    X_imputed = X_numeric.fillna(0)

    non_numeric_cols_remaining = X_imputed.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols_remaining) > 0:
        print(f"  Warning: Dropping non-numeric columns found after selection: {non_numeric_cols_remaining}.")
        X_imputed = X_imputed.drop(columns=non_numeric_cols_remaining)
    return X_imputed, y

if __name__ == '__main__':
    if not LGBM_INSTALLED:
        sys.exit("LightGBM library not found. Please install it (e.g., pip install lightgbm). Exiting.")

    print("--- Running 21_lightgbm_regressor.py (with TimeSeriesSplit HParam tuning & Genre Prediction Output) ---")
    print(f"Config: Max Songs={MAX_SONGS_TO_PROCESS}, Test Fraction={TEST_SET_FRACTION}, Tuning Iterations={N_ITER_RANDOM_SEARCH}, Tuning CV Splits={N_CV_SPLITS_LGBM_TUNING}")

    full_df = load_processed_dataset()
    if full_df.empty:
        print("Loaded dataset is empty. Exiting.")
        sys.exit()
    if 'date' in full_df.columns: # Ensure date is datetime
        full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce')

    all_cols = full_df.columns.tolist()
    identified_genre_cols = [col for col in all_cols if col.startswith('deezer_genre_')]
    identified_dart_cols = [col for col in all_cols if col.startswith('dART_')] # Corrected prefix

    if not identified_genre_cols: print("Warning: No 'deezer_genre_*' columns identified. Genre aggregation will be affected.")
    else: print(f"Identified {len(identified_genre_cols)} Deezer genre columns.")
    if not identified_dart_cols: print("Warning: No 'dART_*' (album record type) columns identified.")
    else: print(f"Identified {len(identified_dart_cols)} dART columns.")

    all_song_predictions_info = []
    song_level_results_lgbm = []
    best_params_log = []

    unique_songs_tuples = full_df[['artist', 'song']].drop_duplicates().values.tolist()
    songs_to_process_list = []
    if MAX_SONGS_TO_PROCESS is not None and 0 < MAX_SONGS_TO_PROCESS < len(unique_songs_tuples):
        print(f"Processing a subset of {MAX_SONGS_TO_PROCESS} songs (selected by most data points).")
        song_counts = full_df.groupby(['artist', 'song']).size().sort_values(ascending=False)
        songs_to_process_list = list(song_counts.head(MAX_SONGS_TO_PROCESS).index)
    else:
        songs_to_process_list = [(r[0], r[1]) for r in unique_songs_tuples]
        print(f"Processing all {len(songs_to_process_list)} unique songs.")

    if not songs_to_process_list:
        print("No songs found to process. Exiting.")
        sys.exit()

    param_dist_lgbm = {
        'n_estimators': randint(50, 250),
        'learning_rate': uniform(0.01, 0.15),
        'num_leaves': randint(20, 60),
        'max_depth': randint(5, 20),
        'reg_alpha': uniform(0, 1.5),
        'reg_lambda': uniform(0, 1.5),
        'colsample_bytree': uniform(0.6, 0.4),
        'subsample': uniform(0.6, 0.4),
        'min_child_samples': randint(5, 30)
    }

    model_name_agg = "LGBM_tuned_ts_cv"
    for i, (artist_name, song_title) in enumerate(tqdm(songs_to_process_list, desc=f"Processing Songs ({model_name_agg})", unit="song")):
        # --- START DEBUG ---
        tqdm.write(f"\nDEBUG: Processing {i+1}/{len(songs_to_process_list)}: {artist_name} - {song_title}")
        # --- END DEBUG ---

        song_df_original = full_df[(full_df['artist'] == artist_name) & (full_df['song'] == song_title)].copy()
        if 'date' in song_df_original.columns:
             song_df_original.sort_values('date', inplace=True)

        min_required_data_lgbm = N_CV_SPLITS_LGBM_TUNING + 5
        # --- START DEBUG ---
        tqdm.write(f"DEBUG: Initial data points: {len(song_df_original)}")
        # --- END DEBUG ---
        if song_df_original.empty or len(song_df_original) < min_required_data_lgbm:
            # --- START DEBUG ---
            tqdm.write(f"DEBUG: Skipping - Insufficient initial data (less than {min_required_data_lgbm}).")
            # --- END DEBUG ---
            continue

        X_song, y_song = prepare_ml_data(song_df_original.copy(), TARGET_COLUMN)
        # --- START DEBUG ---
        tqdm.write(f"DEBUG: After prepare_ml_data - X_song empty: {X_song.empty}, y_song empty: {y_song.empty}, X_song shape: {X_song.shape if not X_song.empty else 'N/A'}")
        # --- END DEBUG ---
        if X_song.empty or X_song.shape[1] == 0 or y_song.empty or y_song.isna().all():
            # --- START DEBUG ---
            tqdm.write(f"DEBUG: Skipping - Empty features/target after preparation.")
            # --- END DEBUG ---
            continue

        # --- Fixed-period split for consistency ---
        N_TEST_PERIODS = 8  # Matches genre-level naive model
        if len(X_song) <= N_TEST_PERIODS:
            tqdm.write(f"DEBUG: Skipping - Not enough data for {N_TEST_PERIODS} test periods.")
            continue

        n_train_points = len(X_song) - N_TEST_PERIODS
        n_test_points = N_TEST_PERIODS

        # --- START DEBUG ---
        tqdm.write(f"DEBUG: Split points - Train: {n_train_points}, Test: {n_test_points}")
        # --- END DEBUG ---
        if n_test_points < 1 or n_train_points < N_CV_SPLITS_LGBM_TUNING + 1:
            # --- START DEBUG ---
            tqdm.write(f"DEBUG: Skipping - Insufficient data for train/test split (Train < {N_CV_SPLITS_LGBM_TUNING + 1}).")
            # --- END DEBUG ---
            continue

        X_train_full, X_test = X_song.iloc[:n_train_points], X_song.iloc[n_train_points:]
        y_train_full, y_test = y_song.iloc[:n_train_points], y_song.iloc[n_train_points:]

        # --- START DEBUG ---
        tqdm.write(f"DEBUG: y_test length: {len(y_test)}. Any non-NaN in y_test: {y_test.notna().any()}")
        if not y_test.notna().any():
             tqdm.write(f"DEBUG: Content of y_test (first 5):\n{y_test.head().to_string()}")
        # --- END DEBUG ---

        test_info_capture_df = song_df_original.iloc[n_train_points:].copy()
        test_info_df_for_genre_agg = pd.DataFrame(index=test_info_capture_df.index)
        if 'date' in test_info_capture_df.columns:
            test_info_df_for_genre_agg['date'] = test_info_capture_df['date']
        else:
            test_info_df_for_genre_agg['date'] = X_test.index

        for gc_col in identified_genre_cols:
            if gc_col in test_info_capture_df.columns:
                test_info_df_for_genre_agg[gc_col] = test_info_capture_df[gc_col]

        test_info_df_for_genre_agg['actual_streams'] = y_test.values
        test_info_df_for_genre_agg['artist'] = artist_name
        test_info_df_for_genre_agg['song'] = song_title

        if X_train_full.empty or y_train_full.empty or X_test.empty or X_train_full.shape[1] == 0:
            tqdm.write(f"DEBUG: Skipping - Empty data after split.")
            continue

        best_lgbm_model = None
        tuning_successful = False
        best_params_found = 'default_fallback'
        cv_score_found = np.nan

        try:
            if len(X_train_full) < N_CV_SPLITS_LGBM_TUNING + 1:
                best_lgbm_model = lgb.LGBMRegressor(objective='regression_l1', metric='mae', n_estimators=100, learning_rate=0.1, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1)
                best_lgbm_model.fit(X_train_full, y_train_full.fillna(0))
                best_params_found = 'default_not_enough_data_for_timeseries_cv'
            else:
                tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS_LGBM_TUNING)
                lgbm_estimator = lgb.LGBMRegressor(objective='regression_l1', metric='mae', random_state=42, n_jobs=1, verbose=-1)

                random_search = RandomizedSearchCV(
                    estimator=lgbm_estimator,
                    param_distributions=param_dist_lgbm,
                    n_iter=N_ITER_RANDOM_SEARCH,
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0,
                    error_score=np.nan
                )
                random_search.fit(X_train_full, y_train_full.fillna(0))

                if not pd.isna(random_search.best_score_):
                    best_lgbm_model = random_search.best_estimator_
                    best_lgbm_model.fit(X_train_full, y_train_full.fillna(0))
                    tuning_successful = True
                    best_params_found = random_search.best_params_
                    cv_score_found = random_search.best_score_
        except ValueError as ve:
            tqdm.write(f"  LGBM TimeSeriesSplit CV Error for {artist_name[:15]}... ({ve}). Using default params.")
        except Exception as e:
            tqdm.write(f"  LGBM Tuning/Fit failed for {artist_name[:15]}...: {e}. Using default params.")

        temp_default_fit_failed = False
        if not tuning_successful and best_lgbm_model is None:
             best_params_found_temp = 'default_fallback_after_exception_or_nan_score'
             try:
                 default_model = lgb.LGBMRegressor(objective='regression_l1', metric='mae', n_estimators=100, learning_rate=0.1, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1)
                 default_model.fit(X_train_full, y_train_full.fillna(0))
                 best_lgbm_model = default_model
                 best_params_found = best_params_found_temp
             except Exception as fit_e:
                  tqdm.write(f"DEBUG: Skipping - Default LGBM fit failed: {fit_e}")
                  temp_default_fit_failed = True
        if temp_default_fit_failed:
            continue

        best_params_log.append({'artist': artist_name, 'song': song_title, 'best_params': str(best_params_found), 'cv_score': cv_score_found})

        try:
            predictions_lgbm = best_lgbm_model.predict(X_test)
            tqdm.write(f"DEBUG: Prediction successful. Number of predictions: {len(predictions_lgbm)}. Any NaNs in predictions: {pd.Series(predictions_lgbm).isna().any()}")
            test_info_df_for_genre_agg[f'predicted_streams_{model_name_agg}'] = predictions_lgbm
            all_song_predictions_info.append(test_info_df_for_genre_agg)
        except Exception as pred_e:
            tqdm.write(f"DEBUG: Skipping - Prediction failed: {pred_e}")
            continue

        predictions_series = pd.Series(predictions_lgbm, index=y_test.index)
        valid_test_indices = y_test.notna() & predictions_series.notna()

        tqdm.write(f"DEBUG: Checking valid_test_indices. Any valid points: {valid_test_indices.any()}")
        if not valid_test_indices.any():
            tqdm.write(f"DEBUG: Skipping - No valid points for evaluation (all test actuals or predictions are NaN, or index mismatch).") # Modified message
            continue

        # Evaluate using the aligned predictions_series
        song_eval_metrics = evaluate_forecast(y_test[valid_test_indices], predictions_series[valid_test_indices], # Use predictions_series here
                                              model_name=f"{model_name_agg} ({artist_name[:15]}...)", print_results=False)
        current_song_eval = {'artist': artist_name, 'song': song_title, **song_eval_metrics}
        if 'model' in current_song_eval: del current_song_eval['model']
        song_level_results_lgbm.append(current_song_eval)

        if len(song_level_results_lgbm) == 1 and ('date' in test_info_df_for_genre_agg.columns):
             tqdm.write(f"  Generating plot for first successful song: {artist_name} - {song_title}")
             try:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(song_df_original['date'].iloc[:n_train_points], y_train_full, label='Train', alpha=0.7)
                ax.plot(test_info_df_for_genre_agg['date'], y_test, label='Test Actual', marker='.')
                # Plot using predictions_series which has the correct index
                ax.plot(test_info_df_for_genre_agg['date'], predictions_series, label=f'{model_name_agg} Forecast', linestyle='--')
                ax.set_title(f"{model_name_agg}: {artist_name} - {song_title}", fontsize=14)
                ax.legend(); ax.grid(True); plt.xticks(rotation=45); plt.tight_layout()
                safe_artist_name = artist_name.replace('/','_').replace('\\','_')
                safe_song_title = song_title.replace('/','_').replace('\\','_')
                plot_filename = FORECAST_PLOT_DIR / f"forecast_plot_{model_name_agg}_{safe_artist_name}_{safe_song_title}.png"
                plt.savefig(plot_filename, bbox_inches='tight')
                plt.close(fig)
             except Exception as plot_err:
                 tqdm.write(f"    Warning: Plot failed for {artist_name} - {song_title}: {plot_err}")
                 if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)

    # --- Genre-Level Aggregation and Evaluation ---
    aggregated_genre_predictions_list = []
    if all_song_predictions_info:
        print(f"\n--- Aggregating {model_name_agg} Predictions for Genre-Level Evaluation ---")
        all_predictions_df_combined = pd.concat(all_song_predictions_info, ignore_index=True)

        if 'date' in all_predictions_df_combined.columns:
            all_predictions_df_combined['date'] = pd.to_datetime(all_predictions_df_combined['date'])

        genre_level_results_agg = []

        if not identified_genre_cols:
            print("No 'deezer_genre_*' columns identified for genre aggregation.")
        else:
            for genre_col_name in tqdm(identified_genre_cols, desc="Aggregating by Genre"):
                if genre_col_name not in all_predictions_df_combined.columns:
                    continue

                genre_specific_predictions_df = all_predictions_df_combined[all_predictions_df_combined[genre_col_name] == 1].copy()
                if genre_specific_predictions_df.empty:
                    continue

                pred_col_to_agg = f'predicted_streams_{model_name_agg}'
                if pred_col_to_agg not in genre_specific_predictions_df.columns:
                    continue

                genre_specific_predictions_df['actual_streams_filled'] = pd.to_numeric(genre_specific_predictions_df['actual_streams'], errors='coerce').fillna(0)
                genre_specific_predictions_df[pred_col_to_agg] = pd.to_numeric(genre_specific_predictions_df[pred_col_to_agg], errors='coerce').fillna(0)

                if 'date' not in genre_specific_predictions_df.columns:
                    continue

                genre_agg_actuals = genre_specific_predictions_df.groupby('date')['actual_streams_filled'].sum()
                genre_agg_predictions = genre_specific_predictions_df.groupby('date')[pred_col_to_agg].sum()

                aligned_genre_df = pd.DataFrame({'actual': genre_agg_actuals, 'predicted': genre_agg_predictions}).dropna()
                if aligned_genre_df.empty or len(aligned_genre_df) < 1:
                    continue

                clean_genre_name = genre_col_name.replace('deezer_genre_', '')
                genre_eval_metrics = evaluate_forecast(aligned_genre_df['actual'], aligned_genre_df['predicted'],
                                                 model_name=f"{model_name_agg} ({clean_genre_name})", print_results=False)

                current_genre_eval_res = {'genre': clean_genre_name, **genre_eval_metrics}
                if 'model' in current_genre_eval_res: del current_genre_eval_res['model']
                genre_level_results_agg.append(current_genre_eval_res)

                genre_agg_df_to_save = aligned_genre_df.reset_index()
                genre_agg_df_to_save['genre'] = clean_genre_name
                genre_agg_df_to_save.rename(columns={'actual': 'actual_streams_total',
                                                     'predicted': 'predicted_streams_total'}, inplace=True)
                aggregated_genre_predictions_list.append(genre_agg_df_to_save[['date','genre','actual_streams_total','predicted_streams_total']])

        if genre_level_results_agg:
            genre_summary_final_df = pd.DataFrame(genre_level_results_agg)
            print(f"\n--- {model_name_agg} - Aggregated Genre-Level Forecast Metrics ---")
            print(genre_summary_final_df.sort_values(by='mape')[['genre','mape','mae','rmse']])

            metrics_filename_genre = FORECAST_CSV_DIR / f"genre_{model_name_agg}_metrics_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
            genre_summary_final_df.to_csv(metrics_filename_genre, index=False)
            print(f"\nAggregated Genre metrics saved to: {metrics_filename_genre.relative_to(project_root)}")

            avg_genre_mape = genre_summary_final_df['mape'].mean(skipna=True)
            if not pd.isna(avg_genre_mape): print(f"Overall Average Genre MAPE ({model_name_agg}): {avg_genre_mape:.2f}%")

        if aggregated_genre_predictions_list:
            combined_genre_predictions_df = pd.concat(aggregated_genre_predictions_list, ignore_index=True)
            predictions_filename_genre = FORECAST_CSV_DIR / f"genre_{model_name_agg}_predictions_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
            combined_genre_predictions_df.to_csv(predictions_filename_genre, index=False)
            print(f"Aggregated genre predictions saved to: {predictions_filename_genre.relative_to(project_root)}")
    else:
        print("\nNo song predictions were made by LGBM, skipping genre aggregation.")

    # --- Song Level Summary ---
    # print(f"\nDEBUG: Checking Song Level Summary section...")
    # print(f"DEBUG: Length of song_level_results_lgbm: {len(song_level_results_lgbm)}")
    # if song_level_results_lgbm:
    #     song_summary_final_df = pd.DataFrame(song_level_results_lgbm)
    #     print(f"\n--- {model_name_agg} - Song-Level Forecast Results Summary (Sample of Top 5 by MAE) ---")
    #     print(song_summary_final_df.sort_values(by='mae').head())

    #     metrics_filename_song = FORECAST_CSV_DIR / f"song_{model_name_agg}_metrics_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
    #     song_summary_final_df.to_csv(metrics_filename_song, index=False)
    #     print(f"\nSong-level metrics saved to: {metrics_filename_song.relative_to(project_root)}")

    #     avg_song_mape = song_summary_final_df['mape'].replace([np.inf, -np.inf], np.nan).mean(skipna=True)
    #     if not pd.isna(avg_song_mape): print(f"Average Song-Level MAPE ({model_name_agg}, excluding inf): {avg_song_mape:.2f}%")

    #     if best_params_log:
    #         params_df = pd.DataFrame(best_params_log)
    #         params_filename = FORECAST_CSV_DIR / f"song_{model_name_agg}_best_params_log_songs{len(songs_to_process_list)}_iter{N_ITER_RANDOM_SEARCH}.csv"
    #         params_df.to_csv(params_filename, index=False)
    #         print(f"Best params log saved to: {params_filename.relative_to(project_root)}")

    #     print(f"DEBUG: Checking condition for saving detailed song predictions...")
    #     print(f"DEBUG: Length of all_song_predictions_info: {len(all_song_predictions_info)}")
    #     if all_song_predictions_info:
    #          print(f"DEBUG: Attempting to concat and save detailed song predictions...")
    #          try:
    #              all_preds_df_to_save = pd.concat(all_song_predictions_info, ignore_index=True)
    #              print(f"DEBUG: Concatenated predictions DataFrame shape: {all_preds_df_to_save.shape}")
    #              preds_filename_song_detail = FORECAST_CSV_DIR / f"song_{model_name_agg}_detailed_predictions_songs{len(songs_to_process_list)}_test{int(TEST_SET_FRACTION * 100)}.csv"
    #              cols_to_save_detail = ['date', 'artist', 'song', 'actual_streams', f'predicted_streams_{model_name_agg}'] + identified_genre_cols
    #              cols_to_save_detail_existing = [col for col in cols_to_save_detail if col in all_preds_df_to_save.columns]
    #              print(f"DEBUG: Columns to save: {cols_to_save_detail_existing}")
    #              all_preds_df_to_save[cols_to_save_detail_existing].to_csv(preds_filename_song_detail, index=False)
    #              print(f"Detailed song predictions saved to: {preds_filename_song_detail.relative_to(project_root)}")
    #          except Exception as e_save:
    #              print(f"--- ERROR during saving detailed song predictions: {e_save} ---")
    #     else:
    #         print("DEBUG: Skipping detailed song predictions save because all_song_predictions_info is empty.")

    # else:
    #     print("DEBUG: Skipping Song Level Summary section because song_level_results_lgbm is empty.")

    print(f"\n--- 21_lightgbm_regressor.py ({model_name_agg}) finished ---")

