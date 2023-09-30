import logging

from pathlib import Path
from typing import List, Dict, Optional

from llama_index.schema import Document

from multi_markdown_reader import MultiMarkdownReader


logger = logging.getLogger(__name__)

def process_and_get_header_docs(folder_path: Path, extra_info: Optional[Dict] = None) -> List[Document]:
    """
    Process markdown files to extract "header-documents."

    Parameters:
        folder_path (Path): Path to the folder containing markdown files.
        extra_info (Optional[Dict]): Additional metadata to include.

    Returns:
        list: List of processed "header-documents."
    """
    multi_markdown_reader = MultiMarkdownReader()
    documents = multi_markdown_reader.load_data_from_folder(folder_path, extra_info=extra_info)
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
