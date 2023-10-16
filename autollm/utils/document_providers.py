from pathlib import Path
from typing import List, Optional, Sequence

from llama_index.schema import Document

from autollm.utils.document_reading import read_files_as_documents
from autollm.utils.git_utils import clone_or_pull_repository


def github_document_provider(git_repo_url: str, local_repo_path: Path,
                             relative_docs_path: Path) -> Sequence[Document]:
    """
    A document provider that fetches documents from a GitHub repository.

    Parameters:
        git_repo_url (str): The URL of the GitHub repository.
        local_repo_path (Path): The local path where the repo will be cloned or updated.
        relative_docs_path (Path): The relative path from the repo root to the docs.
        read_as_single_doc (bool): Whether to treat each file as a single document.

    Returns:
        documents (Sequence[Document]): A sequence of Document objects.
    """
    # Step 1: Clone or pull the GitHub repository to get the latest docs
    clone_or_pull_repository(git_repo_url, local_repo_path)

    # Step 2: Specify the path to the docs
    docs_path = local_repo_path / relative_docs_path

    # Step 3: Read and process the documents
    documents = read_files_as_documents(input_dir=str(docs_path))

    return documents


def local_document_provider(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        *args,
        **kwargs) -> Sequence[Document]:
    """
    A document provider that fetches documents from a local directory.

    Parameters:
        input_dir (str): The path to the directory containing the documents.
        input_files (List): A list of file paths.

    Returns:
        documents (Sequence[Document]): A sequence of Document objects.
    """
    # Step 1: Read and process the documents
    documents = read_files_as_documents(input_dir=input_dir, input_files=input_files, *args, **kwargs)

    return documents
