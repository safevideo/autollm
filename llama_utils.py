# Desc: Utility functions for llama index.
from llama_index import VectorStoreIndex
from typing import List, Type

from markdown_reader import MarkdownReader

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