import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from llama_index.schema import Document

from autollm.utils.multimarkdown_reader import MultiMarkdownReader

logger = logging.getLogger(__name__)


# TODO: add all files supports beside md, use SimpleDirReader
def read_files_as_documents(
        path_or_files: Union[Path, List[Path]],
        read_as_single_doc: bool = False,
        extra_info: Optional[Dict] = None) -> List[Document]:
    """
    Process markdown files to extract documents.

    This function can operate in two modes:
    1. By default (`read_as_single_doc=False`), it extracts "header-documents," where each markdown header defines a new Document.
    2. If `read_as_single_doc=True`, it treats each markdown file as a single Document.

    Parameters:
        path_or_files (Union[Path, List[Path]]): Path to the folder or list of file paths containing markdown files.
        read_as_single_doc (bool): Flag to read the entire markdown file as a single Document.
        extra_info (Optional[Dict]): Additional metadata to include.

    Returns:
        list: List of processed Documents.
    """
    multi_markdown_reader = MultiMarkdownReader(read_as_single_doc=read_as_single_doc)

    # If path_or_files is a Path, check if it is a folder or a file.
    if isinstance(path_or_files, Path):
        if path_or_files.is_dir():
            documents = multi_markdown_reader.load_data_from_folder_or_files(
                folder_path=path_or_files, extra_info=extra_info)
        elif path_or_files.is_file():
            documents = multi_markdown_reader.load_data_from_folder_or_files(
                files=[path_or_files], extra_info=extra_info)
    # If path_or_files is a list of Paths, read all files.
    elif isinstance(path_or_files, list):
        documents = multi_markdown_reader.load_data_from_folder_or_files(
            files=path_or_files, extra_info=extra_info)
    else:
        raise ValueError('Invalid input: path_or_files must be either a Path or a List[Path].')

    logger.info(f"Found {len(documents)} {'header-documents' if not read_as_single_doc else 'documents'}.")
    return documents
