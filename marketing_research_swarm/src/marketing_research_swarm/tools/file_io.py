import pandas as pd

def read_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {e}"

def read_csv(file_path: str):
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        return f"Error reading CSV: {e}"
