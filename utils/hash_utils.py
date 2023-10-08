import hashlib
import logging
from pathlib import Path
from typing import Dict, Sequence, List, Tuple

from llama_index.schema import Document

from vectorstores.base import BaseVS

logger = logging.getLogger(__name__)


def get_md5(file_path: Path) -> str:
    """
    Compute the MD5 hash of a file.

    Parameters:
        file_path (Path): The path to the file.

    Returns:
        str: The MD5 hash of the file.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# TODO: check md5 hashes from vector store index metadata instead of a local txt file (3)
def check_for_changes(documents: Sequence[Document], vs: BaseVS) -> Tuple[Sequence[Document], List[str]]:
    """
    Check for file changes based on their hashes.

    Parameters:
        documents (List[Document]): List of documents to check for changes.
        vs (BaseVS): The vector store to check for changes in.
    
    Returns:
        List[Document]: List of documents that have changed.

        List[str]: List of document ids that are deleted in local but present in vectore store.
    """
    last_hashes, original_file_names, document_ids = vs.get_document_infos()
    current_hashes = {}
    changed_files = []
    deleted_document_ids = []

    markdown_files = [Path(doc.metadata["original_file_path"]) for doc in documents]

    for file in markdown_files:
        current_hash = get_md5(file)
        current_hashes[str(file)] = current_hash
        # Add
        if str(file) not in last_hashes:
            changed_files.append(file)
        # Update
        elif last_hashes[str(file)] != current_hash:
            changed_files.append(file)
        # Remove (file was deleted)
        elif str(file) in last_hashes:
            deleted_document_ids.append(file)

    logger.info(f"Found {len(changed_files)} changed files.")
    logger.info(f"Found {len(deleted_document_ids)} deleted documents.")

    changed_documents = []
    for doc in documents:
        if Path(doc.metadata["original_file_path"]) in changed_files:
            changed_documents.append(doc)

    return changed_documents, deleted_document_ids


def check_for_changes(documents: Sequence[Document], vs: BaseVS) -> Tuple[Sequence[Document], List[str]]:
    """
    Check for file changes based on their hashes.

    Returns:
        List[Document]: List of documents that have changed.

        List[str]: List of document ids that are deleted in local but present in vectore store.
    """
