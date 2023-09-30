# Desc: Utility functions for llama index.
import logging
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from typing import List, Type, Tuple
from pathlib import Path

from markdown_reader import MarkdownReader
from hash_utils import check_for_changes
from markdown_processing import get_markdown_files, process_and_get_header_docs

logger = logging.getLogger(__name__)

def update_index_for_changed_files(index: Type[VectorStoreIndex], files: List[str]):
    """
    Update the index with the changed markdown files.

    This function first deletes all the old documents associated with the changed files
    from the index and then inserts the updated documents.

    Args:
        index (Type[BaseIndex]): The LlamaIndex object to be updated.
        files (List[str]): List of markdown files that have changed.

    Returns:
        None
    """
    # Initialize a MarkdownReader object
    markdown_reader = MarkdownReader()

    # Loop through each file in the list of changed files
    for file in files:
        # Initialize an empty list to store existing doc_ids
        existing_doc_ids = []

        # Iterate over the items in index.ref_doc_info
        for key, value in index.ref_doc_info.items():
            # Check if 'original_file_path' in metadata matches the file path
            if value.metadata.get('original_file_path') == str(file):
                # Append the key (doc_id) to the existing_doc_ids list
                existing_doc_ids.append(key)

        # Delete old documents related to the current file from the index
        for doc_id in existing_doc_ids:
            index.delete_ref_doc(doc_id, delete_from_docstore=True)
 
        # Parse the updated file into a list of Header Documents
        new_documents = markdown_reader.load_data(file)

        # Insert the new documents into the index
        for doc in new_documents:
            index.insert(doc)


def process_and_update_docs(index, base_path, initial_load: bool):
    """Process markdown files and update the Vector Store Index.

    Parameters:
        index (VectorStoreIndex): The Vector Store Index.
        base_path (Path): Base directory to search for markdown files.
    """
    # Get the list of all markdown files in the repository
    markdown_files = get_markdown_files(base_path)

    # If it's not initial load, check for changes
    if not initial_load:
        # Identify files that have changed since the last update
        markdown_files_to_update = check_for_changes(markdown_files)

        # If it's not initial load and there are files to update
        if markdown_files_to_update:
            # Update the index with new documents
            update_index_for_changed_files(index, markdown_files_to_update)
        else:
            logger.info("No changes detected.")



def initialize_or_load_index(docs_path: Path, show_progress: bool = True) -> Tuple[VectorStoreIndex, bool]:
    """
    Initialize or load the Vector Store Index.

    Parameters:
        docs_path (Path): Path to the documents folder.

    Returns:
        VectorStoreIndex: The initialized or loaded index.
        bool: Whether the index was initialized or loaded from disk.
    """
    initial_load = False
    try:
        # Try to load the existing vector store index from disk
        logger.info("Loading existing index.")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        logger.info("Existing index successfully loaded.")
    except FileNotFoundError:
        # If index doesn't exist, create a new one
        logger.info("No existing index found. Creating a new one.")
        documents = process_and_get_header_docs(docs_path)
        index = VectorStoreIndex.from_documents(documents, show_progress=show_progress)
        logger.info("New index successfully created.")
        initial_load = True

    return index, initial_load
