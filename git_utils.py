import subprocess
from pathlib import Path
import os

def clone_and_list_markdown_files(git_repo_url, local_path):
    """
    Clone the git repository and list all markdown files in the 'docs' folder.
    
    Parameters:
        git_repo_url (str): URL of the git repository.
        local_path (str): Local path to clone the repository.
        
    Returns:
        list: List of markdown file paths.
    """
    # Clone or pull the git repository
    if not os.path.exists(local_path):
        subprocess.run(["git", "clone", git_repo_url, local_path])
    else:
        subprocess.run(["git", "-C", local_path, "pull"])
    
    # Identify all markdown files in the 'docs' folder
    docs_path = Path(local_path) / "docs"
    markdown_files = list(docs_path.rglob("*.md"))
    
    return markdown_files