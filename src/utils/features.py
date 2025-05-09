# src/utils/features.py
import pandas as pd
from typing import Sequence, Tuple

def add_time_series_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    metric_col: str = "spotify_streams",
    id_cols: Tuple[str, ...] = ("artist", "song"),
    lag_windows: Sequence[int] = (1, 2, 4, 8),
    roll_windows: Sequence[int] = (4, 8),
) -> pd.DataFrame:
    """
    Enrich a weekly-grain table with calendar, lag and rolling features.

    Parameters
    ----------
    df : DataFrame
        Must contain `date_col`, `metric_col`, and columns in `id_cols`.
    date_col : str
        Column with week-ending timestamps (any Pandas-parseable format).
    metric_col : str
        The numeric series you plan to forecast (e.g. weekly stream count).
    id_cols : Tuple[str, ...]
        Columns that uniquely identify a series (track, artist, region, …).
    lag_windows : list[int]
        How many past weeks to shift for lag features.
    roll_windows : list[int]
        Rolling window sizes (in weeks) for mean/std calculations.

    Returns
    -------
    DataFrame
        Original columns + new calendar, lag and rolling statistics.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # --- Calendar parts -----------------------------------------------------
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Extract calendar features
    df["year"]         = df[date_col].dt.year
    df["month"]        = df[date_col].dt.month
    df["quarter"]      = df[date_col].dt.quarter
    # Use .dt.isocalendar().week for ISO week number, ensure it's integer
    df["weekofyear"]   = df[date_col].dt.isocalendar().week.astype(int)

    # --- Series identifier (temporary) --------------------------------------
    # This temporary ID is used for grouping operations to ensure correctness
    # when calculating lags and rolling statistics per unique time series.
    tmp_id = "__series_id__"
    
    # Create a temporary DataFrame with just the id_cols, converted to string and NaNs filled
    # Ensure id_cols is a list for .loc to select multiple columns correctly
    temp_id_df = df.loc[:, list(id_cols)].fillna("").astype(str)
    
    # Apply the join operation row-wise using apply with a lambda function.
    # This should robustly produce a pandas Series where each element is the joined string.
    joined_ids_series = temp_id_df.apply(lambda row: " - ".join(row), axis=1)
    
    # Assign the resulting Series (after stripping whitespace) to the new tmp_id column
    df[tmp_id] = joined_ids_series.str.strip()

    # Make sure metric col has no NaNs so rolling stats stay numeric.
    # Filling with 0 is a common approach, but consider if another strategy is better
    # (e.g., interpolation, or leaving as NaN if models can handle it).
    if metric_col in df.columns:
        df[metric_col] = df[metric_col].fillna(0)
    else:
        raise KeyError(f"Metric column '{metric_col}' not found in DataFrame.")

    # Sort by the temporary series ID and then by date.
    # This is crucial for .shift() and .rolling() to work correctly.
    df = df.sort_values([tmp_id, date_col])

    # --- Lag features -------------------------------------------------------
    # Create lagged versions of the metric_col for each specified window.
    for k in lag_windows:
        df[f"{metric_col}_lag_{k}"] = (
            df.groupby(tmp_id, observed=True)[metric_col].shift(k) # observed=True can be more efficient
        )

    # --- Rolling window statistics -----------------------------------------
    # Calculate rolling mean and standard deviation for the metric_col.
    for w in roll_windows:
        # Group by the temporary series ID
        grp = df.groupby(tmp_id, observed=True)[metric_col]
        
        # Calculate rolling mean
        # min_periods=1 ensures that a value is produced even if the window is not full (e.g., at the start of a series)
        df[f"{metric_col}_roll_mean_{w}"] = (
            grp.transform(lambda x: x.rolling(window=w, min_periods=1).mean())
        )
        # Calculate rolling standard deviation
        df[f"{metric_col}_roll_std_{w}"] = (
            grp.transform(lambda x: x.rolling(window=w, min_periods=1).std())
        )

    # Clean up by removing the temporary series identifier column
    return df.drop(columns=tmp_id)
