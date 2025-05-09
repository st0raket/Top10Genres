# src/etl/forecasting/14_genre_naive_model.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import warnings
from tqdm import tqdm 

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
    from etl.forecasting.common.evaluation import evaluate_forecast
    import config 
except ModuleNotFoundError as e:
    print(f"Failed to import common modules or config: {e}"); raise

AGGREGATED_GENRE_DATA_DIR = config.PROCESSED_DIR / "genre_aggregated_timeseries"
# --- Define Output Directories ---
FORECAST_CSV_DIR = project_root / "outputs" / "forecasts"
FORECAST_PLOT_DIR = project_root / "outputs" / "plots" /"model"
FORECAST_CSV_DIR.mkdir(parents=True, exist_ok=True) 
FORECAST_PLOT_DIR.mkdir(parents=True, exist_ok=True) 
# --- End Output Directories ---

# --- Configuration for this run ---
N_TEST_PERIODS = 8          # Using 4 weeks test period
TOP_N_GENRES_TO_PROCESS = 10 # Processing Top 5 genres
# --- End Configuration ---

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    if not AGGREGATED_GENRE_DATA_DIR.exists(): sys.exit(f"Directory not found: {AGGREGATED_GENRE_DATA_DIR}. Run 05_aggregate_genre_data.py first.")
    
    print("--- Running 14_genre_naive_model.py ---")
    print(f"Config: Top {TOP_N_GENRES_TO_PROCESS} genres, Test Periods: {N_TEST_PERIODS}")
    print(f"Results will be saved to '{FORECAST_CSV_DIR.relative_to(project_root)}'")
    print(f"Plots will be saved to '{FORECAST_PLOT_DIR.relative_to(project_root)}'")
    
    # (Code for Determining Top N Genres remains the same)
    all_genre_files = sorted(list(AGGREGATED_GENRE_DATA_DIR.glob("genre_ts_*.csv")))
    if not all_genre_files: sys.exit(f"No aggregated genre CSV files found.")
    genre_stream_totals = []
    print("\nRanking genres by total streams...")
    for genre_file_path in tqdm(all_genre_files, desc="Scanning genre files", unit="file", leave=False):
        genre_name_from_file = genre_file_path.stem.replace('genre_ts_', '')
        target_col_in_file = f"{genre_name_from_file}_total_streams"
        try:
            df = pd.read_csv(genre_file_path, usecols=['date', target_col_in_file]) 
            total_streams = df[target_col_in_file].sum()
            genre_stream_totals.append({'genre_name': genre_name_from_file, 'total_streams': total_streams, 'file_path': genre_file_path})
        except Exception as e: print(f"\nWarning: Could not process {genre_file_path} for ranking: {e}")
    if not genre_stream_totals: sys.exit("Could not calculate total streams.")
    sorted_genres = sorted(genre_stream_totals, key=lambda x: x['total_streams'], reverse=True)
    actual_n_to_process = min(TOP_N_GENRES_TO_PROCESS, len(sorted_genres))
    if actual_n_to_process == 0: sys.exit(f"\nNo valid genres found.")
    genre_files_to_process = [g['file_path'] for g in sorted_genres[:actual_n_to_process]]
    print(f"\nTop {len(genre_files_to_process)} genres selected:")
    for i, g_info in enumerate(sorted_genres[:actual_n_to_process]): print(f"  {i+1}. {g_info['genre_name']}")
    # --- End Determine Top N Genres ---

    all_genre_results_naive = []
    all_genre_predictions_list = [] 
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Main Loop: Process Top N Genres ---
    print("\nStarting Naive modeling for selected genres...")
    for genre_file_path in tqdm(genre_files_to_process, desc=f"Processing Top {actual_n_to_process} Genre(s)", unit="genre"):
        genre_name_from_file = genre_file_path.stem.replace('genre_ts_', '')
        target_col_in_file = f"{genre_name_from_file}_total_streams"
        model_type_str = f"Naive_T{N_TEST_PERIODS}" # Unique model type string
        tqdm.write(f"\n--- Processing {model_type_str} for Genre: {genre_name_from_file} ---")

        # Load and Prepare Data
        try:
            genre_ts_df = pd.read_csv(genre_file_path, index_col='date', parse_dates=True)
            if target_col_in_file not in genre_ts_df.columns: tqdm.write(f"  Target column missing. Skipping."); continue
            inferred_freq = pd.infer_freq(genre_ts_df.index) if genre_ts_df.index.is_monotonic_increasing else 'W'; inferred_freq = inferred_freq or 'W'
            genre_ts_df = genre_ts_df[~genre_ts_df.index.duplicated(keep='first')]
            genre_series = genre_ts_df[target_col_in_file].resample(inferred_freq).sum().fillna(0).dropna() 
        except Exception as e: tqdm.write(f"  Error loading/resampling: {e}. Skipping."); continue
        
        # Need at least 1 training point and N test points
        if len(genre_series) < N_TEST_PERIODS + 1: tqdm.write(f"  Not enough data ({len(genre_series)}). Skipping."); continue
        
        # Split data
        train_series = genre_series[:-N_TEST_PERIODS]; test_series = genre_series[-N_TEST_PERIODS:]
        tqdm.write(f"  Train len: {len(train_series)}, Test len: {len(test_series)}")
        if train_series.empty: tqdm.write(f"  Training series empty. Skipping."); continue

        # --- Naive Forecast ---
        last_train_value = train_series.iloc[-1]
        predictions_naive = pd.Series([last_train_value] * len(test_series), index=test_series.index)
        # --- End Naive Forecast ---
        
        # Evaluate
        eval_metrics = evaluate_forecast(test_series, predictions_naive, 
                                         model_name=f"Naive (Genre: {genre_name_from_file})", print_results=False) 
        tqdm.write(f"  Metrics for {genre_name_from_file}: MAE={eval_metrics['mae']:.0f}, MAPE={eval_metrics['mape']:.2f}%")

        # Store results
        current_genre_results = {'genre': genre_name_from_file, 'model': model_type_str, **eval_metrics}; del current_genre_results['model']
        all_genre_results_naive.append(current_genre_results)
        
        # Store predictions
        preds_df = pd.DataFrame({'date': test_series.index, 'genre': genre_name_from_file, 'actual_streams': test_series.values, 'predicted_streams_naive': predictions_naive.values})
        all_genre_predictions_list.append(preds_df)
        
        # --- Plotting (Only first genre) --- 
        if len(all_genre_results_naive) == 1 : 
            try: 
                fig, ax = plt.subplots(figsize=(12, 6))
                # Plot some recent history from train for context
                plot_train = train_series.tail(52) # Show last year of training
                ax.plot(plot_train.index, plot_train, label='Train Data (Recent)', color='dodgerblue')
                ax.plot(test_series.index, test_series, label='Test Data (Actual)', color='darkorange', marker='.')
                ax.plot(predictions_naive.index, predictions_naive, label=f'Naive Forecast', color='limegreen', linestyle='--', marker='x')
                ax.set_title(f"Naive Forecast: Genre {genre_name_from_file}", fontsize=16)
                ax.legend(); ax.grid(True); plt.xticks(rotation=45); plt.tight_layout()
                
                plot_filename = FORECAST_PLOT_DIR / f"forecast_plot_Naive_{genre_name_from_file}.png"
                plt.savefig(plot_filename, bbox_inches='tight')
                tqdm.write(f"  Plot saved to {plot_filename.relative_to(project_root)}")
                plt.show() 
            except Exception as plot_err: tqdm.write(f"  Warning: Plot failed: {plot_err}")
            finally:
                if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        # --- End Plotting ---
            
    # --- Final Summary and Saving Results ---
    if all_genre_results_naive:
        results_summary_df = pd.DataFrame(all_genre_results_naive)
        print(f"\n--- Naive Forecast Results Summary (Top {len(all_genre_results_naive)} Genre(s)) ---")
        print(results_summary_df.sort_values(by='mape')) 
        metrics_filename = FORECAST_CSV_DIR / f"genre_naive_metrics_top{actual_n_to_process}_test{N_TEST_PERIODS}.csv"
        results_summary_df.to_csv(metrics_filename, index=False)
        print(f"\nMetrics summary saved to: {metrics_filename.relative_to(project_root)}")
        avg_mape = results_summary_df['mape'].mean(skipna=True)
        if not pd.isna(avg_mape): print(f"Overall Average MAPE (Naive): {avg_mape:.2f}%")

        if all_genre_predictions_list:
            combined_predictions_df = pd.concat(all_genre_predictions_list, ignore_index=True)
            predictions_filename = FORECAST_CSV_DIR / f"genre_naive_predictions_top{actual_n_to_process}_test{N_TEST_PERIODS}.csv"
            combined_predictions_df.to_csv(predictions_filename, index=False)
            print(f"Predictions saved to: {predictions_filename.relative_to(project_root)}")
    else:
        print(f"\nNo Naive forecast results to summarize or save.")
    print(f"\n--- 14_genre_naive_model.py finished ---")