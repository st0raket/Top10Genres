import pandas as pd
import numpy as np
from pathlib import Path
from dtaidistance import dtw

# Example configuration
input_file = "outputs/forecasts/combined/combined_model_predictions.csv"
output_file = "outputs/forecasts/combined/dtw_distances.csv"

# Define top genres and model to column mappings
top_genres = [
    "pop", "rap_hip_hop", "latin_music", "alternative", "dance",
    "randb", "rock", "electro", "asian_music", "films_games"
]

model_column_mapping = {
    "MovingAverage" : "predicted_streams_best_ma",
    "Naive": "predicted_streams_naive",
    "Prophet": "predicted_streams_prophet_best",
    "SARIMA": "predicted_streams_sarima",
    "LightGBM": "predicted_streams_LGBM_tuned_ts_cv",
    "RandomForest": "predicted_streams_RF_tscv_tuned"
}

def get_aligned_actual_predicted_series(df, top_genres, model_column_mapping):
    """
    Extract predicted series for each genre-model pair,
    ignoring missing actual_streams but ensuring predicted values exist.
    """
    series_collection = {}

    for genre in top_genres:
        df_genre = df[df['genre'] == genre].sort_values('date')

        for model, col in model_column_mapping.items():
            if col in df_genre.columns:
                df_model = df_genre[df_genre['model_name_extracted'] == model]
                df_model = df_model.dropna(subset=[col])  # Only require predictions

                if df_model.empty:
                    print(f"Skipping {genre}-{model}: no predicted data.")
                    continue

                # Fill missing actuals with zeros (or another strategy if preferred)
                df_model['actual_streams_filled'] = df_model['actual_streams'].fillna(0)

                actual_array = df_model['actual_streams_filled'].values
                predicted_array = df_model[col].values

                series_collection[(genre, model)] = (actual_array, predicted_array)
            else:
                print(f"Skipping {genre}-{model}: missing column {col}.")
    return series_collection



def compute_and_export_dtw(df, top_genres, model_column_mapping, output_file):
    """
    Compute DTW distances and export results to a CSV.
    """
    series_data = get_aligned_actual_predicted_series(df, top_genres, model_column_mapping)

    results = []
    for (genre, model), (actual, predicted) in series_data.items():
        distance = dtw.distance(actual, predicted)
        results.append({'genre': genre, 'model': model, 'dtw_distance': distance})
        print(f"Computed DTW for {genre} - {model}: {distance:.4f}")

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ DTW distances saved to {output_file}")
    else:
        print("\n❌ No DTW distances computed. Check your data availability.")

if __name__ == "__main__":
    # Load dataset
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
    else:
        df = pd.read_csv(input_file)
        compute_and_export_dtw(df, top_genres, model_column_mapping, output_file)
