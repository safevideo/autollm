import hashlib
import os

def get_md5(file_path):
    """
    Calculate the MD5 hash of a file.
    
    Parameters:
        file_path (Path): Path to the file.
        
    Returns:
        str: MD5 hash of the file.
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def check_for_changes(markdown_files, hash_file="file_hashes.txt"):
    """
    Check for changes in markdown files by comparing MD5 hashes.
    
    Parameters:
        markdown_files (list): List of markdown file paths.
        hash_file (str): Path to the file where hashes are stored.
        
    Returns:
        list: List of changed file paths.
    """
    changed_files = []
    
    # Load old hashes if available
    old_hashes = {}
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            old_hashes = {line.split()[0]: line.split()[1] for line in f.readlines()}
    
    # Calculate and compare new hashes
    new_hashes = {}
    with open(hash_file, 'w') as f:
        for file_path in markdown_files:
            new_hash = get_md5(file_path)
            if old_hashes.get(str(file_path), None) != new_hash:
                changed_files.append(file_path)
            new_hashes[str(file_path)] = new_hash
            f.write(f"{file_path} {new_hash}\n")
            
    return changed_files
