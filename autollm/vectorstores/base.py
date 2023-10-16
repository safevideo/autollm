from typing import List, Sequence, Tuple, Union

from llama_index import StorageContext, VectorStoreIndex
from llama_index.schema import Document
from llama_index.storage.docstore.types import RefDocInfo
from llama_index.vector_stores.types import BasePydanticVectorStore


# TODO: we need to create separate update_vectorindex, overwrite_vectorindex, delete_documents_by_id functions and use the original llama_index classes for Vector Stores instead of BaseVS
class BaseVS:
    """Base class for vector stores."""

    def __init__(self, in_memory: bool = False):
        self._in_memory = in_memory
        self._vectorstore: Union[BasePydanticVectorStore, VectorStoreIndex] = None

        self._validate_requirements()

    @property
    def vectorstore(self) -> Union[BasePydanticVectorStore, VectorStoreIndex]:
        """
        Get the vector store.

        Returns:
            Union[BasePydanticVectorStore, VectorStoreIndex]: Vector store.
        """
        if self._vectorstore is None:
            raise ValueError('Vector store not connected. Please connect first using connect_vectorstore().')
        return self._vectorstore

    @property
    def vectorindex(self) -> VectorStoreIndex:
        """
        Create a vector store index from the vector store.

        Returns:
            VectorStoreIndex: Vector store index.
        """
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vectorstore)
        # Create index
        if self._in_memory:
            return self.vectorstore
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vectorstore, storage_context=storage_context)

    def _validate_requirements(self):
        """Validate all required env variables are present, and all required packages are installed."""
        raise NotImplementedError

    def update_vectorindex(self, documents: Sequence[Document]):
        """
        Update the vector store index with new documents.

        Parameters:
            documents (Sequence[Document]): List of documents to update.

        Returns:
            None
        """
        for document in documents:
            self.delete_documents_by_id([document.id_])
            self.vectorindex.insert(document)

    def overwrite_vectorindex(self, documents: Sequence[Document]):
        """
        Overwrite the vector store index with new documents.

        Parameters:
            documents (Sequence[Document]): List of documents to overwrite.

        Returns:
            None
        """
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vectorstore)

        # create index, which will insert documents/vectors to vector store
        _ = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def delete_documents_by_id(self, document_ids: Sequence[str]):
        """
        Delete documents from vector store by their ids.

        Parameters:
            document_ids (Sequence[str]): List of document ids to delete.

        Returns:
            None
        """
        # Check if there are any document IDs to delete.
        if not document_ids:
            return

        # Proceed with deletion.
        for document_id in document_ids:
            self.vectorindex.delete_ref_doc(document_id, delete_from_docstore=True)

    def get_document_infos(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Get hashes, original file names, and document ids of all documents in vector store.

        Returns:
            List[str]: hashes,
            List[str]: original_file_names,
            List[str]: document_ids
        """
        hashes, original_file_names, document_ids = [], [], []
        # Retrieve a dict mapping of documents and their nodes+metadata
        for doc_id, vector_object in self.vectorindex.ref_doc_info():
            vector_object: RefDocInfo
            md5_hash = vector_object.metadata.get('md5_hash')
            original_file_name = vector_object.metadata.get('original_file_path')

            hashes.append(md5_hash)
            original_file_names.append(original_file_name)
            document_ids.append(doc_id)

        return hashes, original_file_names, document_ids

    def initialize_vectorindex(self):
        """Create a new vector store index."""
        raise NotImplementedError

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index.

        Sets self._vectorstore.
        """
        raise NotImplementedError
