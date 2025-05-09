# src/etl/forecasting/common/evaluation.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd # Added for potential future use with results dataframes

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles cases where y_true might be zero to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero: filter out true zeros
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask): # All true values are zero
        return np.nan # Or 0, depending on how you want to define MAPE for all-zero true values

    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def evaluate_forecast(y_true, y_pred, model_name="Model", print_results=True):
    """
    Calculates and optionally prints MAE, RMSE, and MAPE for a forecast.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        model_name (str): Name of the model for display purposes.
        print_results (bool): Whether to print the evaluation metrics.
        
    Returns:
        dict: A dictionary containing MAE, RMSE, and MAPE.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same.")
    if len(y_true) == 0:
        print(f"Warning: Empty y_true and y_pred for {model_name}. Returning NaNs for metrics.")
        return {'model': model_name, 'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    
    if print_results:
        print(f"--- {model_name} Evaluation ---")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        if pd.isna(mape):
            print("MAPE: N/A (likely all true values were zero or data was empty)")
        else:
            print(f"MAPE: {mape:.2f}%")
        print("-------------------------")
        
    return {'model': model_name, 'mae': mae, 'rmse': rmse, 'mape': mape}

if __name__ == '__main__':
    print("--- Testing evaluation.py ---")
    
    # Test case 1: Basic scenario
    true_values1 = [100, 110, 120, 105, 90, 115]
    predicted_values1 = [102, 108, 123, 100, 95, 112]
    print("\nTest Case 1: Basic")
    results1 = evaluate_forecast(true_values1, predicted_values1, model_name="Test Model 1")
    print(f"Returned results: {results1}")

    # Test case 2: With zeros in true values
    true_values2 = [10, 0, 20, 0, 30]
    predicted_values2 = [12, 1, 18, -1, 28]
    print("\nTest Case 2: Zeros in true values")
    results2 = evaluate_forecast(true_values2, predicted_values2, model_name="Test Model 2 (with zeros)")
    print(f"Returned results: {results2}")

    # Test case 3: All true values are zero
    true_values3 = [0, 0, 0]
    predicted_values3 = [1, 0, -1]
    print("\nTest Case 3: All true values are zero")
    results3 = evaluate_forecast(true_values3, predicted_values3, model_name="Test Model 3 (all zeros true)")
    print(f"Returned results: {results3}")

    # Test case 4: Empty arrays
    true_values4 = []
    predicted_values4 = []
    print("\nTest Case 4: Empty arrays")
    results4 = evaluate_forecast(true_values4, predicted_values4, model_name="Test Model 4 (empty)")
    print(f"Returned results: {results4}")
    
    # Test case 5: Perfect forecast
    true_values5 = [50, 60, 70]
    predicted_values5 = [50, 60, 70]
    print("\nTest Case 5: Perfect forecast")
    results5 = evaluate_forecast(true_values5, predicted_values5, model_name="Test Model 5 (perfect)")
    print(f"Returned results: {results5}")

    print("\n--- evaluation.py testing finished ---")