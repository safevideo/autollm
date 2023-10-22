from typing import Optional, Sequence

from llama_index import Document, StorageContext, VectorStoreIndex


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


# TODO: add from_config (e.g., from yaml file)
class AutoVectorStoreIndex:
    """AutoVectorStoreIndex lets you dynamically initialize any Vector Store index based on the vector store
    class name and additional parameters.
    """

    @staticmethod
    def from_defaults(
            vector_store_type: str = "LanceDBVectorStore",
            documents: Optional[Sequence[Document]] = None,
            **kwargs) -> VectorStoreIndex:
        """
        Initializes a Vector Store index from Vector Store type and additional parameters.

        Parameters:
            vector_store_type (str): The class name of the vector store (e.g., 'LanceDBVectorStore', 'SimpleVectorStore'..)
            documents (Optional[Sequence[Document]]): Documents to initialize the vector store index from.
            **kwargs: Additional parameters for initializing the vector store

        Returns:
            index (VectorStoreIndex): The initialized Vector Store index instance for given vector store type and parameter set.
        """
        if documents is None and vector_store_type == "SimpleVectorStore":
            raise ValueError("documents must be provided for SimpleVectorStore")

        # Initialize vector store
        VectorStoreClass = import_vector_store_class(vector_store_type)

        # Initialize vector store index from existing vector store
        if documents is None:
            vector_store = VectorStoreClass(**kwargs)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        # Initialize vector store index from documents
        else:
            if vector_store_type == "LanceDBVectorStore" and "uri" not in kwargs:
                kwargs["uri"] = "/tmp/lancedb"
            vector_store = VectorStoreClass(**kwargs)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents=documents, storage_context=storage_context, show_progress=True)

        return index
