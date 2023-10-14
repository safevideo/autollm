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
            input_files: Optional[List] = None,
            path_or_files_to_md: Optional[Union[Path, List[Path]]] = None,
            read_as_single_doc: Optional[bool] = True,
            show_progress: bool = True):
        self._input_files = input_files
        self._path_or_files_to_md = path_or_files_to_md
        self._read_as_single_doc = read_as_single_doc
        self._show_progress = show_progress
        super().__init__(in_memory=True)

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
        else:  # Mock initialization
            documents = [Document.example()]

        self._vectorstore = VectorStoreIndex.from_documents(
            documents=documents, show_progress=self._show_progress)

    def connect_vectorstore(self):
        """
        Connect to a mock vector store index.

        Sets self._vectorstore.
        """
        document = Document.example()
        self._vectorstore = VectorStoreIndex.from_documents(documents=[document])
