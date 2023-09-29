from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
from llama_index import VectorStoreIndex
from pathlib import Path
from typing import List

def delete_docs_from_changed_files(index: VectorStoreIndex, changed_files: List[Path]):
    """
    Delete all documents in the index that belong to the list of changed files.

    Parameters:
        index (SummaryIndex): The index object.
        changed_files (List[Path]): List of changed markdown files.
    """
    for file in changed_files:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="original_file_path", value=str(file))])
        retriever = index.as_retriever(filters=filters)
        
        # Retrieve the documents to delete
        docs_to_delete = retriever.retrieve("")  
        
        for doc in docs_to_delete:
            index.delete_ref_doc(doc.node.id_, delete_from_docstore=True)
