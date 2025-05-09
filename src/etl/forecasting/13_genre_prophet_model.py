# src/etl/forecasting/13_genre_prophet_model.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import warnings
from tqdm import tqdm 
import itertools # Needed for parameter grid

try:
    from prophet import Prophet
    PROPHET_INSTALLED = True
except ImportError:
    PROPHET_INSTALLED = False
    print("CRITICAL: Prophet not installed.")

# --- Path Setup ---
try:
    project_root = Path(__file__).resolve().parents[3]; src_root = Path(__file__).resolve().parents[2]
except IndexError: project_root = Path.cwd(); src_root = Path.cwd() / "src"
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
if str(src_root) not in sys.path: sys.path.insert(0, str(src_root))
# --- End Path Setup ---

try:
    from etl.forecasting.common.evaluation import evaluate_forecast
    import config 
except ModuleNotFoundError as e: print(f"Failed to import common modules or config: {e}"); raise

AGGREGATED_GENRE_DATA_DIR = config.PROCESSED_DIR / "genre_aggregated_timeseries"
FORECAST_CSV_DIR = project_root / "outputs" / "forecasts"; FORECAST_CSV_DIR.mkdir(parents=True, exist_ok=True) 
FORECAST_PLOT_DIR = project_root / "outputs" / "plots"; FORECAST_PLOT_DIR.mkdir(parents=True, exist_ok=True) 

# --- Configuration ---
N_TEST_PERIODS = 8          
TOP_N_GENRES_TO_PROCESS = 10 # Process Top 10 genres
# --- End Configuration ---

# --- Prophet Tuning Grid (Expanded) ---
param_grid_prophet = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5], # Added 0.5
    'seasonality_prior_scale': [0.1, 1.0, 10.0, 20.0], # Added 0.1
    'seasonality_mode': ['additive', 'multiplicative'] 
}
all_params_prophet = [dict(zip(param_grid_prophet.keys(), v)) for v in itertools.product(*param_grid_prophet.values())]
print(f"Prophet Tuning: Testing {len(all_params_prophet)} parameter combinations per genre.")
# --- End Tuning Grid ---

warnings.filterwarnings("ignore", category=FutureWarning) 

if __name__ == '__main__':
    if not PROPHET_INSTALLED: sys.exit("Prophet library not found."); 
    if not AGGREGATED_GENRE_DATA_DIR.exists(): sys.exit(f"Directory not found: {AGGREGATED_GENRE_DATA_DIR}. Run 05_... first.")
    
    print("--- Running 13_genre_prophet_model.py (Expanded Tuning Grid & All Plots) ---")
    print(f"Config: Top {TOP_N_GENRES_TO_PROCESS} genres, Test Periods: {N_TEST_PERIODS}")
    
    # (Code for Determining Top N Genres remains the same)
    all_genre_files = sorted(list(AGGREGATED_GENRE_DATA_DIR.glob("genre_ts_*.csv")))
    if not all_genre_files: sys.exit(f"No aggregated genre CSV files found.")
    genre_stream_totals = []
    print("\nRanking genres..."); 
    for fpath in tqdm(all_genre_files, desc="Scanning files", unit="file", leave=False):
        gname = fpath.stem.replace('genre_ts_', ''); tcol = f"{gname}_total_streams"
        try: df = pd.read_csv(fpath, usecols=['date', tcol]); genre_stream_totals.append({'genre_name': gname, 'total_streams': df[tcol].sum(), 'file_path': fpath})
        except Exception: pass
    if not genre_stream_totals: sys.exit("Could not rank genres.")
    sorted_genres = sorted(genre_stream_totals, key=lambda x: x['total_streams'], reverse=True)
    actual_n_to_process = min(TOP_N_GENRES_TO_PROCESS, len(sorted_genres)); 
    if actual_n_to_process == 0: sys.exit(f"\nNo valid genres found.")
    genre_files_to_process = [g['file_path'] for g in sorted_genres[:actual_n_to_process]]
    print(f"\nTop {len(genre_files_to_process)} genres selected:"); 
    for i, g in enumerate(sorted_genres[:actual_n_to_process]): print(f"  {i+1}. {g['genre_name']}")
    # --- End Determine Top N Genres ---

    all_best_genre_results_prophet = []
    all_best_genre_predictions_list = [] 
    
    # --- Main Loop: Process Top N Genres ---
    print("\nStarting Prophet modeling and tuning for selected genres...")
    for genre_file_path in tqdm(genre_files_to_process, desc=f"Processing Top {actual_n_to_process} Genre(s)", unit="genre"):
        genre_name_from_file = genre_file_path.stem.replace('genre_ts_', '')
        target_col_in_file = f"{genre_name_from_file}_total_streams"
        tqdm.write(f"\n--- Processing Prophet Tuning for Genre: {genre_name_from_file} ---")

        # (Load and Prepare Data remains the same)
        try:
            genre_ts_df_raw = pd.read_csv(genre_file_path, index_col='date', parse_dates=True)
            if target_col_in_file not in genre_ts_df_raw.columns: tqdm.write(f"  Target missing. Skipping."); continue
            prophet_df = genre_ts_df_raw.reset_index()[['date', target_col_in_file]].rename(columns={'date': 'ds', target_col_in_file: 'y'})
            prophet_df['y'] = prophet_df['y'].fillna(0) 
        except Exception as e: tqdm.write(f"  Error loading/preparing: {e}. Skipping."); continue
        min_len_for_prophet = N_TEST_PERIODS + 52 
        if len(prophet_df) < min_len_for_prophet: tqdm.write(f"  Not enough data ({len(prophet_df)}). Skipping."); continue
        train_df = prophet_df.iloc[:-N_TEST_PERIODS]; test_df = prophet_df.iloc[-N_TEST_PERIODS:]
        tqdm.write(f"  Train len: {len(train_df)}, Test len: {len(test_df)}")
        if train_df.empty or len(train_df) < 2: tqdm.write(f"  Training data too short. Skipping."); continue

        # --- Tune Hyperparameters for this genre ---
        best_params_for_genre = None; best_mape_for_genre = float('inf')
        best_predictions_for_genre = None; best_eval_metrics = {}
        best_model_object = None # Store the best model object for plotting
        best_forecast_df = None  # Store the forecast df from the best model
        
        for params in tqdm(all_params_prophet, desc=f"Tuning {genre_name_from_file[:10]}..", unit="combo", leave=False, position=1):
            try:
                model_prophet = Prophet(**params, interval_width=0.95, weekly_seasonality=True, yearly_seasonality=True) 
                model_prophet.fit(train_df)
                future_df = test_df[['ds']].copy() 
                forecast_df = model_prophet.predict(future_df)
                predictions_aligned = pd.Series(forecast_df['yhat'].values, index=test_df['ds'])
                actual_test_values = test_df['y'] 
                if predictions_aligned.isnull().all(): continue 
                eval_metrics_tune = evaluate_forecast(actual_test_values, predictions_aligned, print_results=False) 
                current_mape = eval_metrics_tune.get('mape', float('inf'))
                if not pd.isna(current_mape) and current_mape < best_mape_for_genre:
                    best_mape_for_genre = current_mape; best_params_for_genre = params
                    best_predictions_for_genre = predictions_aligned; best_eval_metrics = eval_metrics_tune 
                    best_model_object = model_prophet # <<< Store best model object
                    best_forecast_df = forecast_df   # <<< Store best forecast df
            except Exception as tune_e: continue 
        # --- End Hyperparameter Tuning Loop ---

        # --- Store results for the best parameters found ---
        if best_params_for_genre is not None:
            tqdm.write(f"  Best Params for {genre_name_from_file}: {best_params_for_genre}")
            tqdm.write(f"  Best Metrics: MAE={best_eval_metrics['mae']:.0f}, MAPE={best_mape_for_genre:.2f}%")
            model_type_str = f"Prophet_tuned_T{N_TEST_PERIODS}"
            current_genre_results = {'genre': genre_name_from_file, 'model': model_type_str, 'best_params': str(best_params_for_genre), **best_eval_metrics}; del current_genre_results['model'] 
            all_best_genre_results_prophet.append(current_genre_results)
            if best_predictions_for_genre is not None:
                 preds_df = pd.DataFrame({'date': test_df['ds'].values, 'genre': genre_name_from_file, 'actual_streams': test_df['y'].values, 'predicted_streams_prophet_best': best_predictions_for_genre.values})
                 all_best_genre_predictions_list.append(preds_df)
                 
            # --- Plotting (Now for ALL processed genres, using best model) --- 
            # Check if we have the best model object and its forecast df
            if best_model_object is not None and best_forecast_df is not None:
                tqdm.write(f"  Generating plots for {genre_name_from_file}...")
                try: 
                    # Plot 1: Forecast vs Actual
                    fig1 = best_model_object.plot(best_forecast_df) # Use stored best model and forecast
                    ax1 = fig1.gca(); ax1.plot(test_df['ds'], test_df['y'], 'r.', markersize=4, label='Actual Test')
                    ax1.set_title(f"Best Prophet Forecast: Genre {genre_name_from_file}"); plt.legend()
                    plot_filename1 = FORECAST_PLOT_DIR / f"forecast_plot_BEST_Prophet_{genre_name_from_file}.png"
                    plt.savefig(plot_filename1, bbox_inches='tight'); plt.close(fig1) # Save & Close
                    tqdm.write(f"    Forecast plot saved: {plot_filename1.name}")

                    # Plot 2: Components
                    fig2 = best_model_object.plot_components(best_forecast_df) # Use stored best model and forecast
                    plot_filename2 = FORECAST_PLOT_DIR / f"components_plot_BEST_Prophet_{genre_name_from_file}.png"
                    fig2.savefig(plot_filename2, bbox_inches='tight'); plt.close(fig2) # Save & Close
                    tqdm.write(f"    Components plot saved: {plot_filename2.name}")
                except Exception as plot_err: 
                    tqdm.write(f"  Warning: Failed to generate/save plot for {genre_name_from_file}: {plot_err}")
                    if 'fig1' in locals() and plt.fignum_exists(fig1.number): plt.close(fig1)
                    if 'fig2' in locals() and plt.fignum_exists(fig2.number): plt.close(fig2)
            else:
                tqdm.write(f"  Skipping plot generation for {genre_name_from_file} as best model/forecast data wasn't stored.")
            # --- End Plotting ---
        else:
            tqdm.write(f"  Prophet tuning did not yield a best result for {genre_name_from_file}.")
            
    # --- Final Summary and Saving Best Results ---
    model_name_save = f"Prophet_tuned_top{actual_n_to_process}_test{N_TEST_PERIODS}_expanded" # Updated filename part
    if all_best_genre_results_prophet:
        # (Saving logic for metrics and predictions CSVs remains the same)
        results_summary_df = pd.DataFrame(all_best_genre_results_prophet)
        print(f"\n--- Best Prophet Forecast Results Summary (Top {len(all_best_genre_results_prophet)} Genre(s)) ---")
        print(results_summary_df.sort_values(by='mape')[['genre','mape','mae','rmse','best_params']]) 
        metrics_filename = FORECAST_CSV_DIR / f"BEST_genre_{model_name_save}_metrics.csv"
        results_summary_df.to_csv(metrics_filename, index=False)
        print(f"\nBest Prophet metrics saved to: {metrics_filename.relative_to(project_root)}")
        avg_mape = results_summary_df['mape'].mean(skipna=True)
        print(f"\nOverall Avg MAPE (Best Prophet): {avg_mape:.2f}%")
        if all_best_genre_predictions_list:
            combined_predictions_df = pd.concat(all_best_genre_predictions_list, ignore_index=True)
            combined_predictions_df.rename(columns={'predicted_streams_prophet_tuned': 'predicted_streams_prophet_best'}, inplace=True)
            predictions_filename = FORECAST_CSV_DIR / f"BEST_genre_{model_name_save}_predictions.csv"
            combined_predictions_df.to_csv(predictions_filename, index=False)
            print(f"Best Prophet predictions saved to: {predictions_filename.relative_to(project_root)}")
    else:
        print(f"\nNo Best Prophet forecast results to summarize or save.")

    print(f"\n--- 13_genre_prophet_model.py (expanded tuning) finished ---")