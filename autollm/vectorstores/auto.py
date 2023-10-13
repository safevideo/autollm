from typing import Any

# Mapping of vector store types to their respective class names.
# This is used for dynamically importing the correct VectorStore class based on the type.
VECTOR_STORE_TYPE_TO_VECTOR_CLASS_NAME = {
    'pinecone': 'PineconeVS',
    'qdrant': 'QdrantVS',
    'inmemory': 'InMemoryVS'
}


def import_vector_store_class(vector_store_type: str, class_name: str):
    """
    Imports a predefined vector store class by class name.

    Args:
        vector_store_type: str
            Type of vector store ("pinecone", "qdrant", "inmemory")
        class_name: str
            Name of the vector store class (example: "PineconeVS")

    Returns:
        The imported VectorStore class.
    """
    module = __import__(f'vectorstores.{vector_store_type}', fromlist=[class_name])
    class_ = getattr(module, class_name)
    return class_


class AutoVectorStore:
    """A class for dynamically initializing a Vector Store based on the type and additional parameters."""

    @staticmethod
    def from_vector_store_type(vector_store_type: str, **kwargs: Any):
        """
        Initializes a Vector Store based on the type and additional parameters.

        Args:
            vector_store_type: str
                Type of vector store ("pinecone", "qdrant")
            **kwargs: Additional parameters for initializing the vector store

        Returns:
            An instance of the appropriate vector store class.

        Raises:
            ValueError: If the given {vector_store_type} is not supported
        """
        try:
            vector_store_class_name = VECTOR_STORE_TYPE_TO_VECTOR_CLASS_NAME[vector_store_type]
        except KeyError:
            raise ValueError(
                f'Invalid store_type: {vector_store_type}. Supported types are {list(VECTOR_STORE_TYPE_TO_VECTOR_CLASS_NAME.keys())}'
            )

        VectorStore = import_vector_store_class(vector_store_type, vector_store_class_name)

        return VectorStore(**kwargs)
