from llama_index.vector_stores.types import BasePydanticVectorStore


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


class AutoVectorStore:
    """A class for dynamically initializing a Vector Store based on its class name and additional
    parameters.
    """

    @staticmethod
    def from_defaults(vector_store_class_name: str, *args, **kwargs) -> BasePydanticVectorStore:
        """
        Initializes a Vector Store based on its class name and additional parameters.

        Args:
            vector_store_class_name: str
                The class name of the vector store (e.g., 'PineconeVectorStore')
            *args: Additional positional arguments for initializing the vector store
            **kwargs: Additional parameters for initializing the vector store

        Returns:
            An instance of the appropriate vector store class.
        """

        VectorStoreClass = import_vector_store_class(vector_store_class_name)

        return VectorStoreClass(*args, **kwargs)
