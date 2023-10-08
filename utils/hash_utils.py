from collections import Counter
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


def check_for_changes(documents: Sequence[Document], vs: BaseVS) -> Tuple[Sequence[Document], List[str]]:
    """
    Check for file changes based on their hashes.

    Parameters:
        documents (Sequence[Document]): List of documents to check for changes.
        vs (BaseVS): The vector store to check for changes in.

    Returns:
        changed_documents (Sequence[Document]): List of documents that have changed.

        deleted_document_ids (List[str]): List of document ids that are deleted in local but present in vector store.
    """
    last_hashes, original_file_names, document_ids = vs.get_document_infos()
    original_file_count = Counter(original_file_names)
    
    changed_documents = []
    deleted_document_ids = []

    for doc in documents:
        file_path = str(Path(doc.metadata["original_file_path"]))
        current_hash = get_md5(Path(file_path))

        # Add
        if file_path not in original_file_names:
            changed_documents.append(doc)
        # Update
        elif current_hash != last_hashes[original_file_names.index(file_path)]:
            changed_documents.append(doc)

        # Mark as processed (for deletion check later)
        if file_path in original_file_count:
            original_file_count[file_path] -= 1
            if original_file_count[file_path] == 0:
                del original_file_count[file_path]

    # Identify documents that are deleted locally but still present in the vector store.
    for remaining_file in original_file_count.keys():
        # Find all indices where this remaining_file appears in original_file_names.
        indices_of_remaining_file = [
            index for index, file_name in enumerate(original_file_names) if file_name == remaining_file
        ]
        
        # Retrieve the document IDs corresponding to these indices.
        corresponding_document_ids = [document_ids[index] for index in indices_of_remaining_file]
        
        # Extend the list of deleted_document_ids with these IDs.
        deleted_document_ids.extend(corresponding_document_ids)

    logger.info(f"Found {len(changed_documents)} changed documents.")
    logger.info(f"Found {len(deleted_document_ids)} locally deleted documents still present in vector store.")

    return changed_documents, deleted_document_ids
