import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS, SimpleDirectoryReader
from llama_index.schema import Document

from autollm.utils.git_utils import clone_or_pull_repository
from autollm.utils.multimarkdown_reader import MultiMarkdownReader
from autollm.utils.pdf_reader import LangchainPDFReader

logger = logging.getLogger(__name__)


def read_files_as_documents(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        filename_as_id: bool = True,
        recursive: bool = True,
        required_exts: Optional[List[str]] = None,
        read_as_single_doc: bool = True,
        **kwargs) -> Sequence[Document]:
    """
    Process markdown files to extract documents using SimpleDirectoryReader.

    Parameters:
        input_dir (str): Path to the directory containing the markdown files.
        input_files (List): List of file paths.
        filename_as_id (bool): Whether to use the filename as the document id.
        recursive (bool): Whether to recursively search for files in the input directory.
        required_exts (Optional[List[str]]): List of file extensions to be read. Defaults to all supported extensions.
        read_as_single_doc (bool): If True, read each markdown as a single document.

    Returns:
        documents (Sequence[Document]): A sequence of Document objects.
    """
    # Configure file_extractor to use MultiMarkdownReader for md files
    file_extractor = {
        **DEFAULT_FILE_READER_CLS, ".md": MultiMarkdownReader(read_as_single_doc=read_as_single_doc),
        ".pdf": LangchainPDFReader(extract_images=False)
    }

    # Initialize SimpleDirectoryReader
    reader = SimpleDirectoryReader(
        file_extractor=file_extractor,
        input_dir=input_dir,
        input_files=input_files,
        filename_as_id=filename_as_id,
        recursive=recursive,
        required_exts=required_exts,
        **kwargs)

    # Read and process the documents
    documents = reader.load_data()

    logger.info(f"Found {len(documents)} 'documents'.")
    return documents


# From http://stackoverflow.com/a/4829285/548792
def on_rm_error(func: Callable, path: str, exc_info: Tuple):
    """
    Error handler for `shutil.rmtree` to handle permission errors.

    Parameters:
        func (Callable): The function that raised the error.
        path (str): The path to the file or directory which couldn't be removed.
        exc_info (Tuple): Exception information returned by sys.exc_info().
    """
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)


def read_github_repo_as_documents(
        git_repo_url: str,
        relative_folder_path: Optional[str] = None,
        required_exts: Optional[List[str]] = None) -> Sequence[Document]:
    """
    A document provider that fetches documents from a specific folder within a GitHub repository.

    Parameters:
        git_repo_url (str): The URL of the GitHub repository.
        relative_folder_path (str, optional): The relative path from the repo root to the folder containing documents.
        required_exts (Optional[List[str]]): List of required extensions.

    Returns:
        Sequence[Document]: A sequence of Document objects.
    """

    # Ensure the temp_dir directory exists
    temp_dir = Path("autollm/temp/")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Temporary directory created at {temp_dir}")

    try:
        # Clone or pull the GitHub repository to get the latest documents
        clone_or_pull_repository(git_repo_url, temp_dir)

        # Specify the path to the documents
        docs_path = temp_dir if relative_folder_path is None else (temp_dir / Path(relative_folder_path))

        # Read and process the documents
        documents = read_files_as_documents(input_dir=str(docs_path), required_exts=required_exts)
        # Logging (assuming logger is configured)
        logger.info(f"Operations complete, deleting temporary directory {temp_dir}..")
    finally:
        # Delete the temporary directory
        shutil.rmtree(temp_dir, onerror=on_rm_error)

    return documents
