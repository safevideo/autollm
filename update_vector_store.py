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
    load_dotenv()

    git_repo_url = os.getenv("GIT_REPO_URL")
    git_repo_path = Path(os.getenv("GIT_REPO_PATH"))
    docs_path = git_repo_path / "docs"

    # Clone or update the repository
    clone_or_pull_repository(git_repo_url, git_repo_path)

    # Load or initialize the index
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    except:
        # Create an index from the documents if it doesn't exist
        documents = process_and_get_header_docs(docs_path)
        index = VectorStoreIndex.from_documents(documents)

    # Get all markdown files
    markdown_files_to_update = get_markdown_files(git_repo_path)

    # Check for file changes
    files_to_update_in_vector_store = check_for_changes(markdown_files_to_update)

    if files_to_update_in_vector_store:
        # Delete old header-docs for changed files
        delete_docs_from_changed_files(index, files_to_update_in_vector_store)

        # Initialize MarkdownReader
        markdown_reader = MarkdownReader()

        # Update the index for changed files
        update_index_for_changed_files(index, files_to_update_in_vector_store, markdown_reader)
    else:
        print("No changes detected.")

    # Persist the index
    index.storage_context.persist()