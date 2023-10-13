from pathlib import Path
from typing import List, Union

from llama_index import VectorStoreIndex

from autollm.utils.markdown_processing import process_and_get_documents
from autollm.vectorstores.base import BaseVS


class InMemoryVS(BaseVS):
    """
    In-memory vector store.

    Loads all documents into memory.
    """

    def __init__(
            self,
            path_or_files: Union[Path, List[Path]],
            read_as_single_doc: bool = True,
            show_progress: bool = True):
        self._path_or_files = path_or_files
        self._read_as_single_doc = read_as_single_doc
        self._show_progress = show_progress
        super().__init__()

    def _validate_requirements(self):
        """For in-memory, no special requirements to validate."""
        pass

    def initialize_vectorindex(self):
        """Create a new vector store index."""
        documents = process_and_get_documents(
            path_or_files=self._path_or_files, read_as_single_doc=self._read_as_single_doc)
        self._vectorstore = VectorStoreIndex.from_documents(
            documents=documents, show_progress=self._show_progress)

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index.

        Sets self._vectorstore.
        """
        # For in-memory, the initialization and connection can be the same.
        self.initialize_vectorindex()
