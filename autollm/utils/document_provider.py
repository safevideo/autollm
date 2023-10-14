from pathlib import Path
from typing import Sequence

from llama_index.schema import Document

from autollm.utils.git_utils import clone_or_pull_repository
from autollm.utils.markdown_processing import process_and_get_documents


# TODO: Use SimpleDirectoryReader and configure its file_extractor argument to use MultiMarkdownReader for md files
def github_document_provider(
        git_repo_url: str,
        local_repo_path: Path,
        relative_docs_path: Path,
        read_as_single_doc: bool = True) -> Sequence[Document]:
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
    documents = process_and_get_documents(docs_path, read_as_single_doc=read_as_single_doc)

    return documents


# TODO: Use SimpleDirectoryReader from llama_index
def local_document_provider(docs_path: Path, read_as_single_doc: bool = True) -> Sequence[Document]:
    """
    A document provider that fetches documents from a local directory.

    Parameters:
        docs_path (Path): The path to the directory containing the docs.
        read_as_single_doc (bool): Whether to treat each file as a single document.

    Returns:
        documents (Sequence[Document]): A sequence of Document objects.
    """
    # Step 1: Read and process the documents
    documents = process_and_get_documents(docs_path, read_as_single_doc=read_as_single_doc)

    return documents
