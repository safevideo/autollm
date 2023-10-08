import logging
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.schema import Document

from .hash_utils import get_md5
from .markdown_reader import MarkdownReader

logger = logging.getLogger(__name__)


def get_markdown_files(base_path: Path) -> List[Path]:
    """
    Get all markdown files in a given path.

    Parameters:
        base_path (Path): Base directory to search for markdown files.

    Returns:
        List[Path]: List of Paths to all markdown files.
    """
    markdown_files = list(base_path.rglob("*.md"))
    logger.info(f"Found {len(markdown_files)} markdown files.")
    return markdown_files


class MultiMarkdownReader(MarkdownReader):
    """MultiMarkdown parser.
    Extract text from multiple markdown files.
    Returns a list of dictionaries with keys as headers and values as the text between headers.
    """

    def __init__(self, *args, read_as_single_doc: bool = False, **kwargs) -> None:
        """Initialize MultiMarkdownReader.

        Parameters:
            read_as_single_doc (bool): If True, read each markdown as a single document.
        """
        super().__init__(*args, **kwargs)
        self.read_as_single_doc = read_as_single_doc

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        content: Optional[str] = None,
    ) -> List[Document]:
        """Include original_file_path in extra_info and respect read_as_single_doc flag."""
        if extra_info is None:
            extra_info = {}

        relative_file_path = str(file)
        extra_info["original_file_path"] = relative_file_path
        extra_info["md5_hash"] = get_md5(file)

        if self.read_as_single_doc:
            # Reading entire markdown as a single document
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            if self._remove_hyperlinks:
                content = self.remove_hyperlinks(content)
            if self._remove_images:
                content = self.remove_images(content)
            # Generate doc_id as file name
            doc_id = relative_file_path

            return [
                Document(
                    id_=doc_id,
                    text=content,
                    metadata=extra_info
                )
            ]
        else:
            # Call parent's load_data method for section-based reading
            return super().load_data(file, extra_info, content)

    def load_data_from_folder_or_files(
            self,
            folder_path: Path = None,
            files: Optional[List[Path]] = None,
            extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Parse all markdown files in a given folder or list of files and return a list of Documents.

        Parameters:
            folder_path (Path): Path to the folder containing markdown files.
            files (Optional[List[Path]]): List of markdown files.
            extra_info (Optional[Dict]): Additional metadata to include.

        Returns:
            List[Document]: List of Documents.
        """
        all_documents = []

        # Gather all markdown files in the folder and its subfolders
        if files:
            all_files = files
        else:
            all_files = get_markdown_files(folder_path)

        for file_path in all_files:
            # Use the overridden load_data method
            documents = self.load_data(file_path, extra_info)
            all_documents.extend(documents)

        return all_documents
