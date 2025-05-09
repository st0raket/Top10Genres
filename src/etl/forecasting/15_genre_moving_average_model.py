# src/etl/forecasting/15_genre_moving_average_model.py

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

# --- Configuration ---
N_TEST_PERIODS = 8          # <<< Set back to 8 based on user request
TOP_N_GENRES_TO_PROCESS = 10 # <<< Set back to 10 based on user request
# <<< List of MA windows to test >>>
MA_WINDOWS_TO_TEST = [2, 3, 4, 6, 8, 12] # Example window sizes
# --- End Configuration ---

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    if not AGGREGATED_GENRE_DATA_DIR.exists(): sys.exit(f"Directory not found: {AGGREGATED_GENRE_DATA_DIR}. Run 05_aggregate_genre_data.py first.")
    
    print("--- Running 15_genre_moving_average_model.py (Find & Save Best Window) ---")
    print(f"Config: Top {TOP_N_GENRES_TO_PROCESS} genres, Test Periods: {N_TEST_PERIODS}")
    print(f"Testing MA Windows: {MA_WINDOWS_TO_TEST}")
    print(f"Results will be saved to '{FORECAST_CSV_DIR.relative_to(project_root)}'")
    # Plots directory mentioned but plot saving is disabled in this version for simplicity
    # print(f"Plots will be saved to '{FORECAST_PLOT_DIR.relative_to(project_root)}'") 
    
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

    # Store results PER WINDOW to find the best later
    all_results_by_window = {} # Key: window_size, Value: list of genre result dicts
    all_predictions_by_window = {} # Key: window_size, Value: list of genre prediction DataFrames

    # <<< Loop through different MA windows to gather results >>>
    print("\nTesting different MA window sizes...")
    for MOVING_AVERAGE_WINDOW in tqdm(MA_WINDOWS_TO_TEST, desc="Testing MA Windows", unit="window"):
        
        current_window_genre_results = []
        current_window_genre_predictions = [] 

        # --- Inner Loop: Process Top N Genres for this window size ---
        for genre_file_path in genre_files_to_process: # No tqdm here, outer loop shows progress
            genre_name_from_file = genre_file_path.stem.replace('genre_ts_', '')
            target_col_in_file = f"{genre_name_from_file}_total_streams"
            model_type_str = f"MA{MOVING_AVERAGE_WINDOW}_T{N_TEST_PERIODS}" 

            # Load and Prepare Data
            try:
                genre_ts_df = pd.read_csv(genre_file_path, index_col='date', parse_dates=True)
                if target_col_in_file not in genre_ts_df.columns: continue 
                inferred_freq = pd.infer_freq(genre_ts_df.index) if genre_ts_df.index.is_monotonic_increasing else 'W'; inferred_freq = inferred_freq or 'W'
                genre_ts_df = genre_ts_df[~genre_ts_df.index.duplicated(keep='first')]
                genre_series = genre_ts_df[target_col_in_file].resample(inferred_freq).sum().fillna(0).dropna() 
            except Exception as e: continue # Skip genre on error
            
            if len(genre_series) < N_TEST_PERIODS + MOVING_AVERAGE_WINDOW: continue 
            
            train_series = genre_series[:-N_TEST_PERIODS]; test_series = genre_series[-N_TEST_PERIODS:]
            if train_series.empty or len(train_series) < MOVING_AVERAGE_WINDOW: continue 

            # --- Moving Average Forecast ---
            ma_value = train_series.iloc[-MOVING_AVERAGE_WINDOW:].mean()
            predictions_ma = pd.Series([ma_value] * len(test_series), index=test_series.index)
            # --- End Moving Average Forecast ---
            
            eval_metrics = evaluate_forecast(test_series, predictions_ma, model_name=f"MA{MOVING_AVERAGE_WINDOW} ({genre_name_from_file})", print_results=False) 
            
            # Store results for this genre & window
            current_genre_results = {'genre': genre_name_from_file, 'ma_window': MOVING_AVERAGE_WINDOW, **eval_metrics}; del current_genre_results['model']
            current_window_genre_results.append(current_genre_results)
            
            # Store predictions for this genre & window
            preds_df = pd.DataFrame({'date': test_series.index, 'genre': genre_name_from_file, 'actual_streams': test_series.values, f'predicted_streams_ma{MOVING_AVERAGE_WINDOW}': predictions_ma.values})
            current_window_genre_predictions.append(preds_df)
                
        # Store results and predictions for the completed window size
        if current_window_genre_results:
            all_results_by_window[MOVING_AVERAGE_WINDOW] = current_window_genre_results
        if current_window_genre_predictions:
            all_predictions_by_window[MOVING_AVERAGE_WINDOW] = current_window_genre_predictions

    # --- Find Best Window and Save Its Results ---
    # ... (Keep the beginning of the script, including the MA_WINDOWS_TO_TEST loop, exactly as before) ...
# ... (The loop collects results in all_results_by_window and all_predictions_by_window) ...

    # --- Find Best Window and Save Its Results ---
    if not all_results_by_window:
        print("\nNo Moving Average results generated for any window size.")
    else:
        print("\n--- Determining Best Moving Average Window ---")
        all_results_list = [item for sublist in all_results_by_window.values() for item in sublist]
        all_results_df = pd.DataFrame(all_results_list)
        
        avg_mape_by_window = all_results_df.groupby('ma_window')['mape'].mean().reset_index()
        print("\nAverage MAPE per Window:")
        print(avg_mape_by_window.sort_values(by='mape'))
        
        best_window_stats = avg_mape_by_window.loc[avg_mape_by_window['mape'].idxmin()]
        BEST_MOVING_AVERAGE_WINDOW = int(best_window_stats['ma_window'])
        best_mape = best_window_stats['mape']
        print(f"\nBest MA Window found: {BEST_MOVING_AVERAGE_WINDOW} (Average MAPE: {best_mape:.2f}%)")

        # --- Save Results for the BEST Window Only ---
        best_window_results_list = all_results_by_window.get(BEST_MOVING_AVERAGE_WINDOW)
        best_window_predictions_list = all_predictions_by_window.get(BEST_MOVING_AVERAGE_WINDOW)

        if best_window_results_list:
            best_results_df = pd.DataFrame(best_window_results_list)
            print(f"\n--- Best MA({BEST_MOVING_AVERAGE_WINDOW}) Forecast Results Summary (Top {len(best_results_df)} Genre(s)) ---")
            print(best_results_df.sort_values(by='mape')) 
            
            metrics_filename = FORECAST_CSV_DIR / f"BEST_genre_ma{BEST_MOVING_AVERAGE_WINDOW}_metrics_top{actual_n_to_process}_test{N_TEST_PERIODS}.csv"
            best_results_df.to_csv(metrics_filename, index=False)
            print(f"\nBest MA metrics saved to: {metrics_filename.relative_to(project_root)}")

            if best_window_predictions_list:
                combined_best_predictions_df = pd.concat(best_window_predictions_list, ignore_index=True)
                pred_col_name_best = f'predicted_streams_ma{BEST_MOVING_AVERAGE_WINDOW}'
                # Ensure the prediction column name exists before renaming
                if pred_col_name_best in combined_best_predictions_df.columns:
                     combined_best_predictions_df.rename(columns={pred_col_name_best: 'predicted_streams_best_ma'}, inplace=True)
                else:
                     # If column name somehow mismatch, try to find it or warn user
                     pred_col_found = [col for col in combined_best_predictions_df.columns if col.startswith('predicted_streams_ma')]
                     if pred_col_found:
                         print(f"Warning: Renaming prediction column '{pred_col_found[0]}' to generic 'predicted_streams_best_ma'")
                         combined_best_predictions_df.rename(columns={pred_col_found[0]: 'predicted_streams_best_ma'}, inplace=True)
                     else:
                          print("Warning: Could not find prediction column to rename for saving.")


                predictions_filename = FORECAST_CSV_DIR / f"BEST_genre_ma{BEST_MOVING_AVERAGE_WINDOW}_predictions_top{actual_n_to_process}_test{N_TEST_PERIODS}.csv"
                combined_best_predictions_df.to_csv(predictions_filename, index=False)
                print(f"Best MA predictions saved to: {predictions_filename.relative_to(project_root)}")

                # <<< NEW: Generate and Save Plot for Best Window (First Genre Processed) >>>
                if not combined_best_predictions_df.empty:
                    # Find the name and predictions for the first genre in the best predictions list
                    first_genre_name_in_best = combined_best_predictions_df['genre'].iloc[0]
                    first_genre_best_preds_df = combined_best_predictions_df[combined_best_predictions_df['genre'] == first_genre_name_in_best]
                    
                    # Find the original file path for this first genre
                    first_genre_info = next((g for g in sorted_genres if g['genre_name'] == first_genre_name_in_best), None)
                    
                    if first_genre_info:
                        print(f"\nGenerating plot for first genre '{first_genre_name_in_best}' using best window {BEST_MOVING_AVERAGE_WINDOW}...")
                        try: 
                            # Reload original data for plotting context
                            genre_ts_df = pd.read_csv(first_genre_info['file_path'], index_col='date', parse_dates=True)
                            target_col = f"{first_genre_name_in_best}_total_streams"
                            inferred_freq = pd.infer_freq(genre_ts_df.index) if genre_ts_df.index.is_monotonic_increasing else 'W'; inferred_freq = inferred_freq or 'W'
                            genre_ts_df = genre_ts_df[~genre_ts_df.index.duplicated(keep='first')]
                            genre_series = genre_ts_df[target_col].resample(inferred_freq).sum().fillna(0).dropna() 
                            train_series = genre_series[:-N_TEST_PERIODS]; test_series = genre_series[-N_TEST_PERIODS:] # Need train/test split again for plot

                            # Ensure indices align before plotting if needed (should align if based on test_series index)
                            predictions_for_plot = first_genre_best_preds_df.set_index('date')['predicted_streams_best_ma']
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            plot_train = train_series.tail(52) # Show last year of training
                            ax.plot(plot_train.index, plot_train, label='Train Data (Recent)', color='dodgerblue')
                            ax.plot(test_series.index, test_series, label='Test Data (Actual)', color='darkorange', marker='.')
                            ax.plot(predictions_for_plot.index, predictions_for_plot, label=f'Best MA({BEST_MOVING_AVERAGE_WINDOW}) Forecast', color='limegreen', linestyle='--', marker='x')
                            ax.set_title(f"Best MA({BEST_MOVING_AVERAGE_WINDOW}) Forecast: Genre {first_genre_name_in_best}", fontsize=16)
                            ax.legend(); ax.grid(True); plt.xticks(rotation=45); plt.tight_layout()
                            
                            # Save the plot
                            plot_filename = FORECAST_PLOT_DIR / f"forecast_plot_BEST_MA{BEST_MOVING_AVERAGE_WINDOW}_{first_genre_name_in_best}.png"
                            plt.savefig(plot_filename, bbox_inches='tight')
                            print(f"Plot for best window saved to: {plot_filename.relative_to(project_root)}")
                            plt.close(fig) # Close plot after saving
                        except Exception as plot_err: 
                            print(f"  Warning: Failed to generate plot for best MA window: {plot_err}")
                            if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig) # Ensure figure closed on error
                    else:
                         print(f"Could not find original data file path to generate plot for genre {first_genre_name_in_best}")
                # <<< End Plotting for Best Window >>>

            else:
                print("Could not find predictions corresponding to the best window.")
        else:
             print("Could not find results corresponding to the best window.")

    print(f"\n--- 15_genre_moving_average_model.py (tuned) finished ---")