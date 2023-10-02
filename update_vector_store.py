import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from git_utils import clone_or_pull_repository
from llama_utils import process_and_update_docs, initialize_or_load_index

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Get environment variables
git_repo_url = os.getenv("GIT_REPO_URL")
git_repo_path = Path(os.getenv("GIT_REPO_PATH"))
# Configure where the markdown files are located
docs_path = git_repo_path / "docs"

def update_vector_store(base_path: Path = docs_path):
    """
    This function is responsible for updating the vector store by performing the following tasks:
    1. Clone or pull the latest repository containing markdown files.
    2. Initialize or load the vector store index.
    3. Check for changed markdown files since the last update.
    4. Delete old header-docs corresponding to changed files from the index.
    5. Insert new header-docs for changed files into the index.
    6. Persist the updated index to disk.

    Parameters:
        base_path (Path): Base directory to search for markdown files.
    """

    # Clone or pull the git repository to get the latest markdown files
    clone_or_pull_repository(git_repo_url, git_repo_path)

    # Initialize or load the vector store index
    index, initial_load = initialize_or_load_index(docs_path=base_path)
        
    # Process markdown files and update the index
    process_and_update_docs(index, base_path, initial_load=initial_load)

    # Persist the updated index to disk
    index.storage_context.persist()

if __name__ == "__main__":
    update_vector_store()