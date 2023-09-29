import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from git_utils import clone_or_pull_repository
from hash_utils import check_for_changes
from llama_utils import delete_docs_from_changed_files, update_index_for_changed_files
from markdown_reader import MarkdownReader
from markdown_processing import process_and_get_header_docs, get_markdown_files

from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage

logging.basicConfig(level=logging.INFO)


def update_vector_store():
    """
    This function is responsible for updating the vector store by performing the following tasks:
    1. Clone or pull the latest repository containing markdown files.
    2. Initialize or load the vector store index.
    3. Check for changed markdown files since the last update.
    4. Delete old header-docs corresponding to changed files from the index.
    5. Insert new header-docs for changed files into the index.
    6. Persist the updated index to disk.
    """
    load_dotenv()

    # Load environment variables for git repository
    git_repo_url = os.getenv("GIT_REPO_URL")
    git_repo_path = Path(os.getenv("GIT_REPO_PATH"))
    docs_path = git_repo_path / "docs"

    # Clone or pull the git repository to get the latest markdown files
    clone_or_pull_repository(git_repo_url, git_repo_path)

    try:
        # Try to load the existing vector store index from disk
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    except:
        # If index doesn't exist, create a new one from available documents
        documents = process_and_get_header_docs(docs_path)
        index = VectorStoreIndex.from_documents(documents)

    # Get the list of all markdown files in the repository
    markdown_files_to_update = get_markdown_files(git_repo_path)

    # Identify files that have changed since the last update
    files_to_update_in_vector_store = check_for_changes(markdown_files_to_update)

    if files_to_update_in_vector_store:
        # Delete outdated documents from the index
        delete_docs_from_changed_files(index, files_to_update_in_vector_store)

        # Initialize the MarkdownReader
        markdown_reader = MarkdownReader()

        # Update the index with new documents
        update_index_for_changed_files(index, files_to_update_in_vector_store, markdown_reader)
    else:
        print("No changes detected for vector store update.")

    # Persist the updated index to disk
    index.storage_context.persist()