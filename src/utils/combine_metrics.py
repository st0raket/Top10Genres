import pandas as pd
from pathlib import Path
import re

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
    elif "ma" in file_name.lower(): model = "MovingAverage"
    return model

def combine_metric_files(input_dir, combined_output):
    """
    Combine all *metrics*.csv files into a single dataset without null genres.
    """
    input_dir = Path(input_dir)
    metric_files = [f for f in input_dir.glob("*metrics*.csv") if "combined" not in f.name]

    combined_dfs = []
    for file in metric_files:
        df = pd.read_csv(file)
        df['model_name_extracted'] = extract_model_metadata(file.name)
        df['source_file'] = file.name
        combined_dfs.append(df)

    if not combined_dfs:
        print("No metric files found.")
        return None

    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Filter out rows with null genres
    combined_df = combined_df[combined_df['genre'].notna()]

    combined_df.to_csv(combined_output, index=False)
    print(f"✅ Full combined dataset saved to {combined_output}")

    return combined_df

def find_best_models_per_genre(df, best_output):
    """
    Find the best model per genre based on RMSE.
    """
    best_per_genre = df.sort_values('rmse').groupby('genre').first().reset_index()
    best_per_genre.to_csv(best_output, index=False)
    print(f"✅ Best models summary saved to {best_output}")
    return best_per_genre

def main():
    input_dir = "outputs/forecasts"
    output_dir = "outputs/forecasts/combined"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    combined_output = Path(output_dir) / "combined_model_metrics_REPORT_READY.csv"
    best_output = Path(output_dir) / "best_models_per_genre.csv"

    combined_df = combine_metric_files(input_dir, combined_output)
    if combined_df is not None:
        find_best_models_per_genre(combined_df, best_output)

if __name__ == "__main__":
    main()
