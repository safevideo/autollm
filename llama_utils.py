from pathlib import Path
from typing import List

from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
from llama_index import VectorStoreIndex, Document


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


def update_index_for_changed_files(index: VectorStoreIndex, changed_files: List[Path], markdown_reader) -> None:
    """
    Delete old documents and insert new ones for the changed files.

    Parameters:
        index (VectorStoreIndex): The current Llama VectorStoreIndex.
        changed_files (List[Path]): List of changed markdown files.
        markdown_reader: An instance of MarkdownReader class.
    """
    # Delete old header-docs for changed files
    delete_docs_from_changed_files(index, changed_files)

    # Process the updated markdown files and insert new header-docs
    for file in changed_files:
        extra_info = {"original_file_path": str(file)}
        new_documents = markdown_reader.load_data(file, extra_info=extra_info)

        for doc in new_documents:
            index.insert(Document(text=doc.text, metadata=doc.metadata))