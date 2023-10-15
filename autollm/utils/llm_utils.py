# Desc: Utility functions for llama index.
import logging
from typing import Sequence

from llama_index import Document
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole

from autollm.auto.vector_store import AutoVectorStore
from autollm.utils.constants import DEFAULT_INDEX_NAME, DEFAULT_VECTORE_STORE_TYPE
from autollm.utils.hash_utils import check_for_changes
from autollm.utils.templates import QUERY_PROMPT_TEMPLATE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def initialize_database(
        documents: Sequence[Document], vectore_store_type: str = DEFAULT_VECTORE_STORE_TYPE) -> None:
    """
    Initializes the vector database for the first time from given documents.

    Parameters:
        documents (Sequence[Document]): List of documents to initialize the vector store with.
        vectore_store_type (str): Type of vector store to use ('qdrant', 'pinecone', etc.).

    Returns:
        None
    """
    logger.info('Initializing vector store')

    # Create a new index and connect to it
    vector_store = AutoVectorStore.from_defaults(
        vector_store_type=vectore_store_type, collection_name=DEFAULT_INDEX_NAME)
    vector_store.initialize_vectorindex()
    vector_store.connect_vectorstore()

    logger.info('Updating vector store with documents')

    # Update the index with the documents
    vector_store.overwrite_vectorindex(documents)

    logger.info('Vector database successfully initialized.')


def update_database(documents: Sequence[Document], vectore_store_type: str) -> None:
    """
    Update the vector database to synchronize it with the provided list of documents.

    This function performs the following actions:
    1. Updates or adds new documents in the vector database that match the input list.
    2. Removes any documents from the vector database that are not present in the input list.

    Parameters:
        documents (Sequence[Document]): Complete set of documents that should exist in the vector database after the update.
        vectore_store_type (str): Specifies the type of vector store to use (e.g., 'qdrant', 'pinecone'). Defaults to DEFAULT_VECTORE_STORE_TYPE.

    Returns:
        None

    Note:
        Ensure that the 'documents' list includes all documents that should remain in the database, as any missing items will be deleted.
    """
    logger.info('Updating vector store')

    # Get changed document ids using the hash of the documents available in the vector store index item metadata
    vector_store = AutoVectorStore.from_defaults(
        vector_store_type=vectore_store_type, index_name=DEFAULT_INDEX_NAME)
    changed_documents, deleted_document_ids = check_for_changes(documents, vector_store)

    # Update the index with the changed documents
    vector_store.update_vectorindex(changed_documents)
    vector_store.delete_documents_by_id(deleted_document_ids)

    logger.info('Vector database successfully updated.')


def set_default_prompt_template() -> ChatPromptTemplate:
    """
    Sets the default prompt template for the query engine.

    Returns:
        SystemPrompt (str): The default system prompt for the query engine.
        ChatPromptTemplate: The default prompt template for the query engine.
    """
    chat_text_msgs = [
        ChatMessage(
            role=MessageRole.USER,
            content=QUERY_PROMPT_TEMPLATE,
        ),
    ]

    return SYSTEM_PROMPT, ChatPromptTemplate(chat_text_msgs)
