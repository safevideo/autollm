from git_utils import clone_or_pull_repository
from hash_utils import check_for_changes
from markdown_processing import process_markdown_files, get_markdown_files
from pathlib import Path

def main():
    git_repo_url = "https://github.com/ultralytics/ultralytics.git"
    git_repo_path = Path("./ultralytics")
    docs_path = git_repo_path / "docs"

    # Clone or update the repository
    clone_or_pull_repository(git_repo_url, git_repo_path)

    # Get all markdown files
    markdown_files = get_markdown_files(git_repo_path)
    print(f"Number of markdown files: {len(markdown_files)}")

    # Check for file changes
    changed_files = check_for_changes(markdown_files)

    if changed_files:
        print(f"Number of changed files: {len(changed_files)}")
        # Process the updated markdown files
        documents = process_markdown_files(docs_path)

        # TODO: Implement function to update the vector database
        # update_vector_index(documents)
    else:
        print("No changes detected.")

if __name__ == "__main__":
    main()
