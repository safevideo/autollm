from typing import Optional, Sequence

from llama_index import Document, VectorStoreIndex


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
            vector_store_type: str,
            documents: Optional[Sequence[Document]] = None,
            *args,
            **kwargs) -> VectorStoreIndex:
        """
        Initializes a Vector Store index from Vector Store type and additional parameters.

        Parameters:
            vector_store_type (str): The class name of the vector store (e.g., 'PineconeVectorStore', 'VectorStoreIndex')
            documents (Optional[Sequence[Document]]): Documents to initialize in memory vector store index.
            *args: Additional positional arguments for initializing the vector store
            **kwargs: Additional parameters for initializing the vector store

        Returns:
            index (VectorStoreIndex): The initialized Vector Store index instance for given vector store type and parameter set.
        """
        if documents is None:
            documents = [Document.example()]
        if vector_store_type == "VectorStoreIndex":
            index = VectorStoreIndex.from_documents(documents=documents, *args, **kwargs)
        else:
            VectorStoreClass = import_vector_store_class(vector_store_type)
            vector_store = VectorStoreClass(*args, **kwargs)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, *args, **kwargs)

        return index
