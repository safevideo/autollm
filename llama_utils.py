# Desc: Utility functions for llama index.
import logging

import pinecone
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import PineconeVectorStore

from pathlib import Path
from typing import List, Type, Union

from env_utils import read_env_variable, validate_environment_variables
from git_utils import clone_or_pull_repository
from hash_utils import check_for_changes
from markdown_processing import get_markdown_files, process_and_get_documents
from multi_markdown_reader import MultiMarkdownReader

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
    # Initialize a MultiMarkdownReader object
    markdown_reader = MultiMarkdownReader()

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


def process_and_update_docs(index: VectorStoreIndex, docs_path: Path):
    """Process markdown files and update the Vector Store Index.

    Parameters:
        index (VectorStoreIndex): The Vector Store Index.
        docs_path (Path): Base directory to search for markdown files.
    """
    # Get the list of all markdown files in the repository
    markdown_files = get_markdown_files(docs_path)

    # Identify files that have changed since the last update
    markdown_files_to_update = check_for_changes(markdown_files)

    # If it's not initial load and there are files to update
    if markdown_files_to_update:
        # Update the index with new documents
        update_index_for_changed_files(index, markdown_files_to_update)
    else:
        logger.info("No changes detected.")


def initialize_database(index_name: str, docs_path: Path, read_as_single_doc: bool) -> None:
    """
    Initialize the database with documents from the specified directory path.
    
    This function initializes the database by reading the document data from a
    given directory path and storing it in a vector database. The function
    uses Pinecone to manage the vector database.
    
    Parameters:
        index_name (str): The name of the Pinecone index_name to load.
        docs_path (Path): Base directory to search for markdown files.
        read_as_single_doc (bool): Flag to read entire markdown as a single document.
        
    Returns:
        None: This function returns None and is used for its side effects.
    """
    logger.info("Initializing the Pinecone database.")

    required_env_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    validate_environment_variables(required_env_vars)

    # Read environment variables for Pinecone initialization
    api_key = read_env_variable("PINECONE_API_KEY")
    environment = read_env_variable("PINECONE_ENVIRONMENT")

    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=environment)
    # Dimensions are for text-embedding-ada-002
    pinecone.create_index(
        "quickstart",
        dimension=1536,
        metric="euclidean",
        pod_type="p1"
    )
    index = pinecone.Index(index_name)

    # Construct vector store
    vector_store = PineconeVectorStore(pinecone_index=index)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Process documents and load them into Pinecone Index
    documents = process_and_get_documents(docs_path, read_as_single_doc=read_as_single_doc)

    # create index, which will insert documents/vectors to pinecone
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return index


def update_database(
    git_repo_url: str, 
    git_repo_path: Path, 
    index_name: str,
    docs_path: Path
) -> None:
    """
    Updates the vector database by performing the following tasks:
    1. Clone or pull the latest repository containing markdown files.
    2. Connect to the the existing vector store database.
    3. Update the index with changed markdown files.

    Parameters:
        git_repo_url (str): URL of the git repository to clone or pull.
        git_repo_path (Path): Local path to clone the git repository.
        index_name (str): The name of the Pinecone index_name to load.
        docs_path (Path): Base directory to search for markdown files.

    Returns:
        None
    """
    logger.info("Starting to update the vector database.")

    # Step 1: Clone or pull the git repository to get the latest markdown files
    clone_or_pull_repository(git_repo_url, git_repo_path)

    # Step 2: Load the existing vector store index into memory
    connect_database(index_name=index_name)

    # Step 3: Update the index with changed markdown files
    process_and_update_docs(index=index_name, docs_path=docs_path)

    logger.info("Vector database successfully updated.")


def connect_database(index_name: str) -> Union[VectorStoreIndex, None]:
    """
    Conntect to existing database with data already loaded in.

    Parameters:
        index_name (str): The name of the Pinecone index to connect to.

    Returns:
        VectorStoreIndex: The loaded vector store index.
        None: If the index could not be loaded.

    Raises:
        Exception: Detailed exception information if the index fails to load.
    """
    required_env_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    validate_environment_variables(required_env_vars)

    api_key = read_env_variable("PINECONE_API_KEY")
    environment = read_env_variable("PINECONE_ENVIRONMENT")

    try:
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create an index instance that targets the given index_name
        pinecone_index = pinecone.Index(index_name)
        
        # Initialize an instance of the Pinecone vector store module
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # Connect to the vectore store with the data already loaded in
        loaded_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        logger.info(f"Successfully connected to the database with index: {index_name}")
        
        return loaded_index

    except Exception as e:
        logger.error(f"Failed to load Pinecone vector store index: {index_name}. Error: {e}")
        return None
