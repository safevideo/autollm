from llama_index import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext

from .base import BaseVS
from utils.markdown_processing import process_and_get_documents

class InMemoryVectorStore(BaseVS):
    def __init__(self, use_async: bool = False, show_progress: bool = False):
        self._use_async = use_async
        self._show_progress = show_progress
        super().__init__()

    def _validate_requirements(self):
        """
        For in-memory, no special requirements to validate.
        """
        pass

    def initialize_vectorindex(self):
        """
        Create a new vector store index.
        """
        documents = process_and_get_documents(path_or_files="README.md", read_as_single_doc=True)
        self._vectorstore = VectorStoreIndex.from_documents(
            documents=documents,
            show_progress=self._show_progress
        )

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index. Sets self._vectorstore.
        """
        # For in-memory vector stores, "connecting" doesn't make sense as there is no external
        # resource to connect to. In this context, you might just want to "use" the existing vector index.
        self._vectorstore = self.vectorindex

