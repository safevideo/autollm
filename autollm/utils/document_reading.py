import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.schema import Document

from autollm.utils.git_utils import clone_or_pull_repository
from autollm.utils.multimarkdown_reader import MultiMarkdownReader

logger = logging.getLogger(__name__)


def read_files_as_documents(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        filename_as_id: bool = True,
        recursive: bool = True,
        read_as_single_doc: bool = True,
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
        **kwargs)

    # Read and process the documents
    documents = reader.load_data()

    logger.info(f"Found {len(documents)} 'documents'.")
    return documents


def read_github_repo_as_documents(git_repo_url: str,
                                  relative_folder_path: Optional[str] = None) -> Sequence[Document]:
    """
    A document provider that fetches documents from a specific folder within a GitHub repository.

    Parameters:
        git_repo_url (str): The URL of the GitHub repository.
        relative_folder_path (str): The relative path from the repo root to the folder containing documents.

    Returns:
        documents (Sequence[Document]): A sequence of Document objects.
    """
    # Step 1: Create a temporary directory for cloning the repository
    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        print(temp_dir)
        # Step 2: Clone or pull the GitHub repository to get the latest documents
        clone_or_pull_repository(git_repo_url, Path(temp_dir))

        # Step 3: Specify the path to the documents
        if relative_folder_path is None:
            docs_path = temp_dir
        else:
            docs_path = temp_dir / Path(relative_folder_path)

        # Step 4: Read and process the documents
        documents = read_files_as_documents(input_dir=str(docs_path))
        logger.info(f"Deleting temporary directory {temp_dir}..")

    # Step 5: The temporary directory will be deleted upon exiting the 'with' block

    return documents
