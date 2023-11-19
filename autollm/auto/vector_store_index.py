from typing import Optional, Sequence

from llama_index import Document, ServiceContext, StorageContext, VectorStoreIndex
from llama_index.schema import BaseNode


def import_vector_store_class(vector_store_class_name: str):
    """
    Imports a predefined vector store class by class name.

    Args:
    Returns:
        The imported VectorStore class.
    """
    module = __import__("llama_index.vector_stores", fromlist=[vector_store_class_name])
    class_ = getattr(module, vector_store_class_name)
    return class_


class AutoVectorStoreIndex:
    """AutoVectorStoreIndex lets you dynamically initialize any Vector Store index based on the vector store
    class name and additional parameters.
    """

    @staticmethod
    def from_defaults(
            vector_store_type: str = "LanceDBVectorStore",
            lancedb_uri: str = "./.lancedb",
            lancedb_table_name: str = "vectors",
            documents: Optional[Sequence[Document]] = None,
            nodes: Optional[Sequence[BaseNode]] = None,
            service_context: Optional[ServiceContext] = None,
            **kwargs) -> VectorStoreIndex:
        """
        Initializes a Vector Store index from Vector Store type and additional parameters.

        Parameters:
            vector_store_type (str): The class name of the vector store (e.g., 'LanceDBVectorStore', 'SimpleVectorStore'..)
            documents (Optional[Sequence[Document]]): Documents to initialize the vector store index from.
            nodes (Optional[Sequence[BaseNode]]): Nodes to initialize the vector store index from.
            service_context (Optional[ServiceContext]): Service context to initialize the vector store index from.
            **kwargs: Additional parameters for initializing the vector store

        Returns:
            index (VectorStoreIndex): The initialized Vector Store index instance for given vector store type and parameter set.
        """
        if documents is None and nodes is None and vector_store_type == "SimpleVectorStore":
            raise ValueError("documents or nodes must be provided for SimpleVectorStore")

        if documents is not None and nodes is not None:
            raise ValueError("documents and nodes cannot be provided at the same time")

        # Initialize vector store
        VectorStoreClass = import_vector_store_class(vector_store_type)

        # If LanceDBVectorStore, use lancedb_uri and lancedb_table_name
        if vector_store_type == "LanceDBVectorStore":
            vector_store = VectorStoreClass(uri=lancedb_uri, table_name=lancedb_table_name, **kwargs)
        else:
            vector_store = VectorStoreClass(**kwargs)

        # Initialize vector store index from existing vector store
        if documents is None and nodes is None:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, service_context=service_context)
            return index

        # Initialize vector store index from documents or nodes
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if documents is not None:
            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=True)
        else:
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=True)

        return index
