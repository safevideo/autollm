import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from git_utils import clone_or_pull_repository
from hash_utils import check_for_changes
from markdown_processing import process_markdown_files, get_markdown_files

logging.basicConfig(level=logging.INFO)


def main():
    load_dotenv()

    git_repo_url = os.getenv("GIT_REPO_URL")
    git_repo_path = Path(os.getenv("GIT_REPO_PATH"))
    docs_path = git_repo_path / "docs"

    # Clone or update the repository
    clone_or_pull_repository(git_repo_url, git_repo_path)

    # Get all markdown files
    markdown_files = get_markdown_files(git_repo_path)

    # Check for file changes
    changed_files = check_for_changes(markdown_files)

    if changed_files:
        # Process the updated markdown files
        documents = process_markdown_files(docs_path)

        # TODO: Implement function to update the vector database
        # update_vector_index(documents)
    else:
        print("No changes detected.")

if __name__ == "__main__":
    main()
