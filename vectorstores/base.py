from typing import Sequence

from llama_index import StorageContext, VectorStoreIndex
from llama_index.schema import Document
from llama_index.vector_stores.types import BasePydanticVectorStore
from llama_index.storage.docstore.types import RefDocInfo


class BaseVS:
    def __init__(self):
        self._vectorstore: BasePydanticVectorStore = None

        self._validate_requirements()

    @property
    def vectorstore(self) -> BasePydanticVectorStore:
        if self._vectorstore is None:
            raise ValueError("Vector store not connected. Please connect first using connect_vectorstore().")
        return self._vectorstore

    @property
    def vectorindex(self) -> VectorStoreIndex:
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vectorstore)
        # Create index
        return VectorStoreIndex.from_vector_store(vector_store=self.vectorstore, storage_context=storage_context)

    def _validate_requirements(self):
        """
        Validate all required env variables are present, and all required packages are installed.
        """
        raise NotImplementedError

    def update_vectorindex(self, documents: Sequence[Document]):
        for document in documents:
            self.delete_documents_by_id([document.id_])
            self.vectorindex.insert(document)

    def overwrite_vectorindex(self, documents: Sequence[Document]):
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vectorstore)

        # create index, which will insert documents/vectors to vector store
        _ = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def delete_documents_by_id(self, document_ids: Sequence[str]):
        # Delete from vector store
        for document_id in document_ids:
            self.vectorindex.delete_ref_doc(document_id, delete_from_docstore=True)

    def get_document_infos(self):
        """
        Get document infos from vector store.
        
        Returns:
            List[str]: hashes,
            List[str]: original_file_names,
            List[str]: document_ids
        """
        hashes, original_file_names, document_ids = [], [], []
        # Retrieve a dict mapping of documents and their nodes+metadata
        for doc_id, vector_object in self.vectorindex.ref_doc_info():
            vector_object: RefDocInfo
            hash = vector_object.get_document_hash(self, doc_id)
            original_file_name = vector_object.metadata.get('original_file_path')

            hashes.append(hash)
            original_file_names.append(original_file_name)
            document_ids.append(doc_id)

        return hashes, original_file_names, document_ids

    
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
