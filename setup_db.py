from pathlib import Path
import logging
import os

from env_utils import read_env_variable, validate_environment_variables
from git_utils import clone_or_pull_repository
from llama_utils import initialize_database

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_database(index_name: str = "quickstart", read_as_single_doc: bool = True) -> None:
    """
    One-time setup for initializing the vector database with documents.
    
    This function should be executed once locally to populate the vector database.
    """
    required_env_variables = ["DOCS_PATH"]
    validate_environment_variables(required_env_variables)
    try:
        # Get environment variables
        git_repo_url = read_env_variable("GIT_REPO_URL", "https://github.com/ultralytics/ultralytics.git")
        git_repo_path = Path(read_env_variable("GIT_REPO_PATH", "./ultralytics"))
        # Configure where the markdown files are located
        docs_path = read_env_variable("DOCS_PATH")

        # Clone or pull the git repository to get the latest markdown files
        clone_or_pull_repository(git_repo_url, git_repo_path)

        # Setup the database
        logger.info("Starting database setup.")
        initialize_database(index_name, docs_path, read_as_single_doc=read_as_single_doc)
        logger.info("Database setup completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during setup: {e}")
        raise


if __name__ == "__main__":
    setup_database()
