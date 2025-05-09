# 🎵 Top 10 Music Genre Forecasting Capstone

## 📂 Project Structure with Descriptions

Capstone/
├── data/ # Input and processed data
│ ├── raw/ # Original raw scraped data (Spotify, Apple Music)
│ ├── processed/ # Cleaned, enriched, and feature-engineered datasets
│ │ ├── checkpoints/ # Intermediate data checkpoints
│ │ └── genre_aggregated_timeseries/ # Genre-level aggregated time series data
├── notebooks/ # Jupyter notebooks for EDA and experimentation
├── outputs/ # Model outputs and visualizations
│ ├── forecasts/ # Forecast result files (CSV, metrics, predictions)
│ │ └── combined/ # Combined model comparison results
│ └── plots/ # Generated charts and graphs
│   └── EDA/
│   └── model/
│   └── Report/
├── paper/
└── src/ # Source code for ETL and modeling pipelines
  └── etl/ # Data extraction, transformation, and loading
    ├── forecasting/ # Forecasting model implementations
    └── utils/ # Utilities for summarizing and post-processing


## Execution Order

1. **ETL Pipeline**
   - Run all scripts in the `src/etl/` directory, following the numeric order specified in each filename.

2. **Exploratory Data Analysis**
   - Execute the `notebooks/EDA.ipynb` notebook.

3. **Forecasting Pipeline**
   - Run all scripts in the `src/etl/forecasting/` directory, following the numeric order specified in each filename.

4. **Utility Scripts**
   - Execute the `combine_...` scripts located in the `src/utils/` directory.
   - Execute the `src/utils/summarize_dtw.py` script.

5. **Final Reporting**
   - Run the `notebooks/final_report.ipynb` notebook to generate the final analysis and visualizations.
   

## Github link: https://github.com/st0raket/Top10Genres