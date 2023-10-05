from pathlib import Path
import logging
import os

from git_utils import clone_or_pull_repository
from llama_utils import initialize_database

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_environment_variables() -> None:
    """Validate that all required environment variables are set."""
    required_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    for var in required_vars:
        if var not in os.environ:
            raise EnvironmentError(f"Environment variable {var} is not set")


def setup_database(index_name: str = "quickstart", read_as_single_doc: bool = True) -> None:
    """
    One-time setup for initializing the vector database with documents.
    
    This function should be executed once locally to populate the vector database.
    """
    try:
        validate_environment_variables()

        # Get environment variables
        git_repo_url = read_env_variable("GIT_REPO_URL")
        git_repo_path = Path(read_env_variable("GIT_REPO_PATH"))
        # Configure where the markdown files are located
        docs_path = git_repo_path / "docs"

        # Clone or pull the git repository to get the latest markdown files
        clone_or_pull_repository(git_repo_url, git_repo_path)

        docs_path = Path("your_docs_directory_here")  # Replace with your docs directory path

        logger.info("Starting database setup.")
        initialize_database(index_name, docs_path, read_as_single_doc=read_as_single_doc)
        logger.info("Database setup completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during setup: {e}")
        raise


if __name__ == "__main__":
    setup_database()
