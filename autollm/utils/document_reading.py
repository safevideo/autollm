import logging
from typing import List, Optional, Sequence

from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.schema import Document

from autollm.utils.multimarkdown_reader import MultiMarkdownReader

logger = logging.getLogger(__name__)


def read_files_as_documents(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        filename_as_id: bool = True,
        recursive: bool = True,
        read_as_single_doc: bool = True,
        *args,
        **kwargs) -> Sequence[Document]:
    """
    Process markdown files to extract documents using SimpleDirectoryReader.

    Parameters:
        input_dir (str): Path to the directory containing the markdown files.
        input_files (List): List of file paths.
        filename_as_id (bool): Whether to use the filename as the document id.
        recursive (bool): Whether to recursively search for files in the input directory.
        read_as_single_doc (bool): If True, read each markdown as a single document.

    Returns:
        documents (Sequence[Document]): A sequence of Document objects.
    """
    # Configure file_extractor to use MultiMarkdownReader for md files
    file_extractor = {".md": MultiMarkdownReader(read_as_single_doc=read_as_single_doc)}

    # Initialize SimpleDirectoryReader
    reader = SimpleDirectoryReader(
        file_extractor=file_extractor,
        input_dir=input_dir,
        input_files=input_files,
        filename_as_id=filename_as_id,
        recursive=recursive,
        *args,
        **kwargs)

    # Read and process the documents
    documents = reader.load_data()

    logger.info(f"Found {len(documents)} {'header-documents' if not read_as_single_doc else 'documents'}.")
    return documents
