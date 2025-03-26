import pandas as pd

def load_data(file_path="customer_financial_data.csv"):
    """
    Loads customer financial data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)  # Replace with actual dataset path
    return data
