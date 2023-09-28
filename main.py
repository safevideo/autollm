from git_utils import clone_and_list_markdown_files
from hash_utils import check_for_changes
from markdown_processing import process_markdown_files
from pathlib import Path

def main():
    git_repo_url = "https://github.com/ultralytics/ultralytics.git"
    local_path = Path("./ultralytics")

    # Step 1: Clone and list markdown files
    markdown_files = clone_and_list_markdown_files(git_repo_url, local_path)
    print(f"Number of markdown files: {len(markdown_files)}")
    
    # Step 2: Check for file changes
    changed_files = check_for_changes(markdown_files)

    if changed_files:
        # Step 3: Process the updated markdown files
        documents = process_markdown_files(local_path / "docs")

        # TODO: Implement function to update the vector database
        # Step 4: Update the vector database
        # update_vector_index(documents)
    else:
        print("No changes detected.")

if __name__ == "__main__":
    main()
