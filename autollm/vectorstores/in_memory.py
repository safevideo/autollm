from pathlib import Path
from typing import List, Optional, Union

from llama_index import Document, SimpleDirectoryReader, VectorStoreIndex

from autollm.utils.markdown_processing import process_and_get_documents
from autollm.vectorstores.base import BaseVS


class InMemoryVS(BaseVS):
    """
    In-memory vector store.

    Loads all documents into memory.
    """

    def __init__(
            self,
            mock_initialization: bool = False,
            input_files: List = None,
            path_or_files_to_md: Optional[Union[Path, List[Path]]] = None,
            read_as_single_doc: bool = True,
            show_progress: bool = True):
        self._mock_initialization = mock_initialization
        self._input_files = input_files
        self._path_or_files_to_md = path_or_files_to_md
        self._read_as_single_doc = read_as_single_doc
        self._show_progress = show_progress
        super().__init__()

    def _validate_requirements(self):
        """For in-memory, no special requirements to validate."""
        pass

    def initialize_vectorindex(self):
        """Create a new vector store index."""
        if self._path_or_files_to_md:
            documents = process_and_get_documents(
                path_or_files=self._path_or_files_to_md, read_as_single_doc=self._read_as_single_doc)
        elif self._input_files:
            documents = SimpleDirectoryReader(input_files=self._input_files).load_data()
        elif self._mock_initialization:
            documents = [Document.example()]
        else:
            raise ValueError(
                '`mock_initialization`, `input_files` or `path_or_files_to_md` must be provided.')

        self._vectorstore = VectorStoreIndex.from_documents(
            documents=documents, show_progress=self._show_progress)

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index.

        Sets self._vectorstore.
        """
        document = Document.example()
        self._vectorstore = VectorStoreIndex.from_documents(documents=[document])
