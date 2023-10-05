import logging
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.schema import Document

from .multimarkdown_reader import MultiMarkdownReader

logger = logging.getLogger(__name__)


def process_and_get_documents(folder_path: Path, read_as_single_doc: bool = False, extra_info: Optional[Dict] = None) -> List[Document]:
    """
    Process markdown files to extract documents.

    This function can operate in two modes:
    1. By default (`read_as_single_doc=False`), it extracts "header-documents," where each markdown header defines a new Document.
    2. If `read_as_single_doc=True`, it treats each markdown file as a single Document.

    Parameters:
        folder_path (Path): Path to the folder containing markdown files.
        read_as_single_doc (bool): Flag to read the entire markdown file as a single Document.
        extra_info (Optional[Dict]): Additional metadata to include.

    Returns:
        list: List of processed Documents.
    """
    multi_markdown_reader = MultiMarkdownReader(read_as_single_doc=read_as_single_doc)
    documents = multi_markdown_reader.load_data_from_folder_or_files(folder_path, extra_info=extra_info)
    logger.info(f"Found {len(documents)} {'header-documents' if not read_as_single_doc else 'documents'}.")
    return documents
