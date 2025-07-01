from crewai.tools import BaseTool
from .file_io import read_file, read_csv

class ReadFileTool(BaseTool):
    name: str = "Read File"
    description: str = "Reads the content of a file and returns it as a string."

    def _run(self, file_path: str) -> str:
        return read_file(file_path)

class ReadCSVTool(BaseTool):
    name: str = "Read CSV"
    description: str = "Reads a CSV file and returns its content as a dictionary."

    def _run(self, file_path: str) -> dict:
        df = read_csv(file_path)
        return df.to_dict()

read_file_tool = ReadFileTool()
read_csv_tool = ReadCSVTool()
