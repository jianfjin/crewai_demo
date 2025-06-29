from langchain.tools import tool
from .file_io import read_file, read_csv

@tool("Read File")
def read_file_tool(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    """
    return read_file(file_path)

@tool("Read CSV")
def read_csv_tool(file_path: str):
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    return read_csv(file_path)
