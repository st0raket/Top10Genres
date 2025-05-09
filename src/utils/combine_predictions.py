import pandas as pd
from pathlib import Path

def extract_model_metadata(file_name):
    """
    Extract model name from file name using heuristic rules.
    """
    model = "Unknown"
    if "naive" in file_name.lower(): model = "Naive"
    elif "prophet" in file_name.lower(): model = "Prophet"
    elif "sarima" in file_name.lower(): model = "SARIMA"
    elif "genre_rf_" in file_name.lower(): model = "RandomForest"
    elif "genre_lgbm_" in file_name.lower(): model = "LightGBM"
    elif "ma4" in file_name.lower(): model = "MovingAverage"
    return model

def combine_prediction_files(input_dir, combined_output):
    """
    Combine all *predictions*.csv files into a single dataset.
    """
    input_dir = Path(input_dir)
    prediction_files = [f for f in input_dir.glob("*predictions*.csv") if "combined" not in f.name]

    combined_dfs = []
    for file in prediction_files:
        df = pd.read_csv(file)
        df['model_name_extracted'] = extract_model_metadata(file.name)
        df['source_file'] = file.name
        combined_dfs.append(df)

    if not combined_dfs:
        print("No prediction files found.")
        return None

    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Drop rows where 'genre' is missing or NaN
    before_drop_rows = len(combined_df)
    combined_df = combined_df.dropna(subset=['genre'])
    after_drop_rows = len(combined_df)
    print(f"Dropped {before_drop_rows - after_drop_rows} rows without 'genre'.")

    # Drop columns that are entirely zero or NaN
    non_zero_cols = combined_df.loc[:, (combined_df != 0).any(axis=0)]
    non_nan_cols = non_zero_cols.dropna(axis=1, how='all')
    final_df = non_nan_cols

    # Expand predicted_streams_total to model-specific columns if applicable
    if 'predicted_streams_total' in final_df.columns:
        rf_mask = final_df['model_name_extracted'] == 'RandomForest'
        lgbm_mask = final_df['model_name_extracted'] == 'LightGBM'
        final_df.loc[rf_mask, 'predicted_streams_RF_tscv_tuned'] = final_df.loc[rf_mask, 'predicted_streams_total']
        final_df.loc[lgbm_mask, 'predicted_streams_LGBM_tuned_ts_cv'] = final_df.loc[lgbm_mask, 'predicted_streams_total']

    # Convert date to datetime if it's not already
    final_df['date'] = pd.to_datetime(final_df['date'], errors='coerce')

    # Find the earliest SARIMA date
    sarima_dates = final_df.loc[final_df['model_name_extracted'] == 'SARIMA', 'date']
    if not sarima_dates.empty:
        cutoff_date = sarima_dates.min()
        print(f"Applying cutoff: keeping rows on or after {cutoff_date.date()}")
        
        before_cutoff_count = len(final_df)
        final_df = final_df[final_df['date'] >= cutoff_date]
        after_cutoff_count = len(final_df)
        print(f"Filtered from {before_cutoff_count} to {after_cutoff_count} rows.")
    else:
        print("No SARIMA data found. Skipping date cutoff.")

    # Save final cleaned dataframe
    final_df.to_csv(combined_output, index=False)
    print(f"✅ Full combined predictions saved to {combined_output}")
    return final_df


    

def main():
    input_dir = "outputs/forecasts"
    output_dir = "outputs/forecasts/combined"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    combined_output = Path(output_dir) / "combined_model_predictions.csv"

    combine_prediction_files(input_dir, combined_output)


if __name__ == "__main__":
    main()
