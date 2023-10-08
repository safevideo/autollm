from pathlib import Path
import logging

from utils import(
    env_utils,
    git_utils,
    llm_utils
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_database(index_name: str = "quickstart", read_as_single_doc: bool = True) -> None:
    """
    Perform a one-time setup to initialize the vector database with documents.
    
    This function should be executed once to populate the vector database with initial documents.
    It clones or pulls a Git repository to access the latest markdown files, then initializes the database.
    
    Parameters:
        index_name (str): The name of the Pinecone index to use. Default is 'quickstart'.
        read_as_single_doc (bool): Whether to treat each markdown file as a single document. Default is True.
    """
    required_env_variables = ["DOCS_PATH"]
    env_utils.validate_environment_variables(required_env_variables)

    # Get environment variables
    git_repo_url = env_utils.read_env_variable("GIT_REPO_URL", "https://github.com/ultralytics/ultralytics.git")
    git_repo_path = Path(env_utils.read_env_variable("GIT_REPO_PATH", "./ultralytics"))
    docs_path = env_utils.read_env_variable("DOCS_PATH").lstrip('/') # Remove leading slash if present
    full_path = git_repo_path.joinpath(docs_path)   # Concatenate paths

    # Clone or pull the git repository to get the latest markdown files
    git_utils.clone_or_pull_repository(git_repo_url, git_repo_path)

    # Setup the database
    logger.info("Starting database setup.")
    llm_utils.initialize_database(index_name, full_path, read_as_single_doc=read_as_single_doc)
    logger.info("Database setup completed successfully.")


if __name__ == "__main__":
    setup_database()
