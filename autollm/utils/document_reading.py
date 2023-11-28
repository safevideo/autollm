import os
import shutil
import stat
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.schema import Document

from autollm.utils.git_utils import clone_or_pull_repository
from autollm.utils.logging import logger
from autollm.utils.markdown_reader import MarkdownReader
from autollm.utils.pdf_reader import LangchainPDFReader
from autollm.utils.webpage_reader import WebPageReader
from autollm.utils.website_reader import WebSiteReader


def read_files_as_documents(
        input_dir: Optional[str] = None,
        input_files: Optional[List] = None,
        exclude_hidden: bool = True,
        filename_as_id: bool = True,
        recursive: bool = True,
        required_exts: Optional[List[str]] = None,
        **kwargs) -> Sequence[Document]:
    """
    Process markdown files to extract documents using SimpleDirectoryReader.

    Parameters:
        input_dir (str): Path to the directory containing the markdown files.
        input_files (List): List of file paths.
        exclude_hidden (bool): Whether to exclude hidden files.
        filename_as_id (bool): Whether to use the filename as the document id.
        recursive (bool): Whether to recursively search for files in the input directory.
        required_exts (Optional[List[str]]): List of file extensions to be read. Defaults to all supported extensions.

    Returns:
        documents (Sequence[Document]): A sequence of Document objects.
    """
    # Configure file_extractor to use MarkdownReader for md files
    file_extractor = {
        ".md": MarkdownReader(read_as_single_doc=True),
        ".pdf": LangchainPDFReader(extract_images=False)
    }

    # Initialize SimpleDirectoryReader
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        exclude_hidden=exclude_hidden,
        file_extractor=file_extractor,
        input_files=input_files,
        filename_as_id=filename_as_id,
        recursive=recursive,
        required_exts=required_exts,
        **kwargs)

    logger.info(f"Reading files from {input_dir}..") if input_dir else logger.info(
        f"Reading files {input_files}..")

    # Read and process the documents
    documents = reader.load_data()

    logger.info(f"Found {len(documents)} 'document(s)'.")
    return documents


# From http://stackoverflow.com/a/4829285/548792
def on_rm_error(func: Callable, path: str, exc_info: Tuple):
    """
    Error handler for `shutil.rmtree` to handle permission errors.

    Parameters:
        func (Callable): The function that raised the error.
        path (str): The path to the file or directory which couldn't be removed.
        exc_info (Tuple): Exception information returned by sys.exc_info().
    """
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)


def read_github_repo_as_documents(
        git_repo_url: str,
        relative_folder_path: Optional[str] = None,
        required_exts: Optional[List[str]] = None) -> Sequence[Document]:
    """
    A document provider that fetches documents from a specific folder within a GitHub repository.

    Parameters:
        git_repo_url (str): The URL of the GitHub repository.
        relative_folder_path (str, optional): The relative path from the repo root to the folder containing documents.
        required_exts (Optional[List[str]]): List of required extensions.

    Returns:
        Sequence[Document]: A sequence of Document objects.
    """

    # Ensure the temp_dir directory exists
    temp_dir = Path("autollm/temp/")
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cloning github repo {git_repo_url} into temporary directory {temp_dir}..")

    try:
        # Clone or pull the GitHub repository to get the latest documents
        clone_or_pull_repository(git_repo_url, temp_dir)

        # Specify the path to the documents
        docs_path = temp_dir if relative_folder_path is None else (temp_dir / Path(relative_folder_path))

        # Read and process the documents
        documents = read_files_as_documents(input_dir=str(docs_path), required_exts=required_exts)
        # Logging (assuming logger is configured)
        logger.info(f"Operations complete, deleting temporary directory {temp_dir}..")
    finally:
        # Delete the temporary directory
        shutil.rmtree(temp_dir, onerror=on_rm_error)

    return documents


def read_website_as_documents(
        parent_url: Optional[str] = None,
        sitemap_url: Optional[str] = None,
        include_filter_str: Optional[str] = None,
        exclude_filter_str: Optional[str] = None) -> List[Document]:
    """
    Read documents from a website or a sitemap.

    Parameters:
        parent_url (str, optional): The starting URL from which to scrape documents.
        sitemap_url (str, optional): The URL of the sitemap to process.
        include_filter_str (str, optional): Filter string to include certain URLs.
        exclude_filter_str (str, optional): Filter string to exclude certain URLs.

    Returns:
        List[Document]: A list of Document objects containing content and metadata.

    Raises:
        ValueError: If neither parent_url nor sitemap_url is provided, or if both are provided.
    """
    if (parent_url is None and sitemap_url is None) or (parent_url is not None and sitemap_url is not None):
        raise ValueError("Please provide either parent_url or sitemap_url, not both or none.")

    reader = WebSiteReader()
    if parent_url:
        documents = reader.load_data(
            parent_url=parent_url,
            include_filter_str=include_filter_str,
            exclude_filter_str=exclude_filter_str)
    else:
        documents = reader.load_data(
            sitemap_url=sitemap_url,
            include_filter_str=include_filter_str,
            exclude_filter_str=exclude_filter_str)

    return documents


def read_webpage_as_documents(url: str) -> List[Document]:
    """
    Read documents from a single webpage URL using the WebPageReader.

    Parameters:
        url (str): The URL of the web page to read.

    Returns:
        List[Document]: A list of Document objects containing content and metadata from the web page.
    """
    reader = WebPageReader()
    documents = reader.load_data(url)
    return documents
