from git import Repo
from pathlib import Path

def clone_or_pull_repository(git_url: str, local_path: Path) -> None:
    """
    Clone a Git repository or pull latest changes if it already exists.

    Parameters:
        git_url (str): The URL of the Git repository.
        local_path (Path): The local path where the repository will be cloned or updated.
    """
    if local_path.exists():
        repo = Repo(str(local_path))
        repo.remotes.origin.pull()
    else:
        Repo.clone_from(git_url, str(local_path))
