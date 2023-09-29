import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from git_utils import clone_or_pull_repository
from hash_utils import check_for_changes
from llama_utils import delete_docs_from_changed_files
from markdown_reader import MarkdownReader
from markdown_processing import process_markdown_files, get_markdown_files

from llama_index import VectorStoreIndex, Document

logging.basicConfig(level=logging.INFO)


def main():
    load_dotenv()

    git_repo_url = os.getenv("GIT_REPO_URL")
    git_repo_path = Path(os.getenv("GIT_REPO_PATH"))
    docs_path = git_repo_path / "docs"

    documents = process_markdown_files(docs_path)

    index = VectorStoreIndex.from_documents(documents)

    # Clone or update the repository
    clone_or_pull_repository(git_repo_url, git_repo_path)

    # Get all markdown files
    markdown_files = get_markdown_files(git_repo_path)

    # Check for file changes
    changed_files = check_for_changes(markdown_files)

    if changed_files:
        # Delete old header-docs for changed files
        delete_docs_from_changed_files(index, changed_files)

        # Initialize MarkdownReader
        markdown_reader = MarkdownReader()

        # Process the updated markdown files
        for file in changed_files:
            extra_info = {"original_file_path": str(file)}
            new_documents = markdown_reader.load_data(file, extra_info=extra_info)
            
            # Insert new header-docs into the index
            for doc in new_documents:
                index.insert(Document(text=doc.text, metadata=doc.metadata))
    else:
        print("No changes detected.")

if __name__ == "__main__":
    main()
