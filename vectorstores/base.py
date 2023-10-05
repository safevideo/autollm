from typing import Sequence

from llama_index import StorageContext, VectorStoreIndex
from llama_index.schema import Document
from llama_index.vector_stores.types import BasePydanticVectorStore


class BaseVS:
    def __init__(self):
        self._vectorstore: BasePydanticVectorStore = None

        self._validate_requirements()

    @property
    def vectorstore(self) -> BasePydanticVectorStore:
        if self.vectorstore is None:
            raise ValueError("Vector store not connected. Please connect first using connect_vectorstore().")
        raise self._vectorstore

    @property
    def vectorindex(self) -> VectorStoreIndex:
        if self.vectorstore is None:
            raise ValueError("Vector store not connected. Please connect first using connect_vectorstore().")
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self._vectorstore)
        # Create index
        return VectorStoreIndex.from_vector_store(vector_store=self.vectorstore, storage_context=storage_context)

    def _validate_requirements(self):
        """
        Validate all required env variables are present, and all required packages are installed.
        """
        raise NotImplementedError

    def update_vectorindex(self, documents: Sequence[Document]):
        if self._vectorstore is None:
            raise ValueError("Vector store not connected. Please connect first using connect_vectorstore().")

        for document in documents:
            self.vectorindex.delete(document.id_)
            self.vectorindex.insert(document)

    def overwrite_vectorindex(self, documents: Sequence[Document]):
        if self._vectorstore is None:
            raise ValueError("Vector store not connected. Please connect first using connect_vectorstore().")

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self._vectorstore)

        # create index, which will insert documents/vectors to vector store
        _ = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def initialize_vectorindex(self):
        """
        Create a new vector store index.
        """
        raise NotImplementedError

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index. Sets self._vectorstore.
        """
        raise NotImplementedError