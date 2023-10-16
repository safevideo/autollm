# db_utils.py
import logging
from typing import Sequence

from llama_index import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore, QdrantVectorStore

from autollm.auto.vector_store_index import AutoVectorStoreIndex
from autollm.utils.constants import DEFAULT_INDEX_NAME
from autollm.utils.env_utils import read_env_variable
from autollm.utils.hash_utils import check_for_changes

logger = logging.getLogger(__name__)


def initialize_pinecone_index(
        index_name: str, dimension: int = 1536, metric: str = 'euclidean', pod_type: str = 'p1'):
    import pinecone

    # Read environment variables for Pinecone initialization
    api_key = read_env_variable('PINECONE_API_KEY')
    environment = read_env_variable('PINECONE_ENVIRONMENT')

    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=environment)
    pinecone.create_index(index_name, dimension=dimension, metric=metric, pod_type=pod_type)


def initialize_qdrant_index(index_name: str, size: int = 1536, distance: str = 'EUCLID'):
    """Initialize Qdrant index."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    # Initialize client
    url = read_env_variable('QDRANT_URL')
    api_key = read_env_variable('QDRANT_API_KEY')

    client = QdrantClient(url=url, api_key=api_key)

    # Convert string distance measure to Distance Enum equals to Distance.EUCLID
    distance = Distance[distance]

    # Create index
    client.recreate_collection(
        collection_name=index_name, vectors_config=VectorParams(size=size, distance=distance))


def connect_vectorstore(vector_store, **params):
    """Connect to an existing vector store."""
    import pinecone
    from qdrant_client import QdrantClient

    # Logic to connect to vector store based on the specific type of vector store
    if isinstance(vector_store, PineconeVectorStore):
        vector_store.pinecone_index = pinecone.Index(params['index_name'])
    elif isinstance(vector_store, QdrantVectorStore):
        vector_store.client = QdrantClient(url=params['url'], api_key=params['api_key'])
    # TODO: Add more elif conditions for other vector stores as needed


def update_vector_store_index(vector_store_index: VectorStoreIndex, documents: Sequence[Document]):
    """
    Update the vector store index with new documents.

    Parameters:
        vector_store_index: An instance of AutoVectorStoreIndex or any compatible vector store.
        documents (Sequence[Document]): List of documents to update.

    Returns:
        None
    """
    for document in documents:
        delete_documents_by_id(vector_store_index, [document.id_])
        vector_store_index.insert(document)


def overwrite_vectorindex(vector_store, documents: Sequence[Document]):
    """
    Overwrite the vector store index with new documents.

    Parameters:
        vector_store: An instance of AutoVectorStore or any compatible vector store.
        documents (Sequence[Document]): List of documents to overwrite.

    Returns:
        None
    """
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index, which will insert documents/vectors to vector store
    _ = VectorStoreIndex.from_documents(documents, storage_context=storage_context)


def delete_documents_by_id(vector_store_index: VectorStoreIndex, document_ids: Sequence[str]):
    """
    Delete documents from vector store by their ids.

    Parameters:
        vector_store_index: An instance of AutoVectorStoreIndex or any compatible vector store.
        document_ids (Sequence[str]): List of document ids to delete.

    Returns:
        None
    """
    # Check if there are any document IDs to delete.
    if not document_ids:
        return

    # Proceed with deletion.
    for document_id in document_ids:
        vector_store_index.delete_ref_doc(document_id, delete_from_docstore=True)


# TODO: refactor and update.
# def initialize_database(
#         documents: Sequence[Document], vector_store_class_name: str, **vector_store_params) -> None:
#     logger.info('Initializing vector store')

#     vector_store = AutoVectorStore.from_defaults(vector_store_class_name, **vector_store_params)

#     if vector_store_class_name == 'PineconeVectorStore':
#         initialize_pinecone_index(vector_store, **vector_store_params)
#     elif vector_store_class_name == 'QdrantVectorStore':
#         initialize_qdrant_index(vector_store, **vector_store_params)
#     # TODO: Add more elif conditions for other vector stores as needed

#     connect_vectorstore(vector_store, **vector_store_params)

#     logger.info('Updating vector store with documents')

#     update_vector_store_index(vector_store, documents)

#     logger.info('Vector database successfully initialized.')

# # TODO: refactor and update.
# def update_database(documents: Sequence[Document], vectore_store_type: str) -> None:
#     """
#     Update the vector database to synchronize it with the provided list of documents.

#     This function performs the following actions:
#     1. Updates or adds new documents in the vector database that match the input list.
#     2. Removes any documents from the vector database that are not present in the input list.

#     Parameters:
#         documents (Sequence[Document]): Complete set of documents that should exist in the vector database after the update.
#         vectore_store_type (str): Specifies the type of vector store to use (e.g., 'qdrant', 'pinecone'). Defaults to DEFAULT_VECTORE_STORE_TYPE.

#     Returns:
#         None

#     Note:
#         Ensure that the 'documents' list includes all documents that should remain in the database, as any missing items will be deleted.
#     """
#     logger.info('Updating vector store')

#     # Get changed document ids using the hash of the documents available in the vector store index item metadata
#     vector_store = AutoVectorStore.from_defaults(
#         vector_store_type=vectore_store_type, index_name=DEFAULT_INDEX_NAME)
#     changed_documents, deleted_document_ids = check_for_changes(documents, vector_store)

#     # Update the index with the changed documents
#     vector_store.update_vectorindex(changed_documents)
#     vector_store.delete_documents_by_id(deleted_document_ids)

#     logger.info('Vector database successfully updated.')
