import hashlib
import logging
from pathlib import Path
from typing import Dict, Sequence

from llama_index.schema import Document

logger = logging.getLogger(__name__)


def get_md5(file_path: Path) -> str:
    """
    Compute the MD5 hash of a file.

    Parameters:
        file_path (Path): The path to the file.

    Returns:
        str: The MD5 hash of the file.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_last_hashes(hash_file: Path) -> Dict[str, str]:
    """
    Load the last known hashes from a file.

    Parameters:
        hash_file (Path): The path to the hash file.

    Returns:
        Dict[str, str]: A dictionary mapping file paths to their MD5 hashes.
    """
    if hash_file.exists():
        with open(hash_file, 'r') as f:
            return {line.split()[0]: line.split()[1] for line in f.readlines()}
    return {}


def save_current_hashes(current_hashes: Dict[str, str], hash_file: Path) -> None:
    """
    Save the current hashes to a file.

    Parameters:
        current_hashes (Dict[str, str]): A dictionary mapping file paths to their MD5 hashes.
        hash_file (Path): The path to the hash file.
    """
    with open(hash_file, 'w') as f:
        for file, hash in current_hashes.items():
            f.write(f"{file} {hash}\n")


# TODO: check md5 hashes from vector store index metadata instead of a local txt file
def check_for_changes(documents: Sequence[Document], hash_file: Path = Path("file_hashes.txt")) -> Sequence[Document]:
    """
    Check for file changes based on their MD5 hashes.

    Parameters:
        documents (List[Document]): List of documents to check for changes.
        hash_file (Path): The path to the hash file.

    Returns:
        List[Document]: List of documents that have changed.
    """
    last_hashes = load_last_hashes(hash_file)
    current_hashes = {}
    changed_files = []

    markdown_files = [Path(doc.metadata["original_file_path"]) for doc in documents]

    for file in markdown_files:
        current_hash = get_md5(file)
        current_hashes[str(file)] = current_hash
        if str(file) not in last_hashes or last_hashes[str(file)] != current_hash:
            changed_files.append(file)

    logger.info(f"Found {len(changed_files)} changed files.")

    save_current_hashes(current_hashes, hash_file)

    changed_documents = []
    for doc in documents:
        if Path(doc.metadata["original_file_path"]) in changed_files:
            changed_documents.append(doc)

    return changed_documents
