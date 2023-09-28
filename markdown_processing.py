from multi_markdown_reader import MultiMarkdownReader
from pathlib import Path

def process_markdown_files(folder_path: Path) -> list:
    """
    Process markdown files to extract "header-documents."

    Parameters:
        folder_path (Path): Path to the folder containing markdown files.

    Returns:
        list: List of processed "header-documents."
    """
    markdown_reader = MultiMarkdownReader()
    documents = markdown_reader.load_data_from_folder(folder_path)
    return documents