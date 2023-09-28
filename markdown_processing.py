import logging
from multi_markdown_reader import MultiMarkdownReader
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

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


def get_markdown_files(repo_path: Path, docs_folder: Path = Path("docs")) -> List[Path]:
    """
    Get all markdown files in the docs folder of a Git repository.

    Parameters:
        repo_path (Path): The path to the Git repository.
        docs_folder (Path): The path to the docs folder within the repository. Defaults to "docs".

    Returns:
        List[Path]: List of Paths to all markdown files.
    """
    docs_path = repo_path / docs_folder
    markdown_files = list(docs_path.glob('**/*.md'))
    logger.info(f"Found {len(markdown_files)} markdown files.")
    return markdown_files
