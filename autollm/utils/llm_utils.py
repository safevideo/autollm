# Desc: Utility functions for llama index.
import logging
from typing import Optional, Sequence, Union

import tiktoken
from llama_index import Document, ServiceContext, set_global_service_context
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import Anyscale, OpenAI, PaLM
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole

from autollm.auto.vector_store import AutoVectorStore
from autollm.utils.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_ENABLE_TOKEN_COUNTING,
    DEFAULT_INDEX_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_VECTORE_STORE_TYPE,
    MODEL_COST,
)
from autollm.utils.env_utils import read_env_variable
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

    # Connect to the existing vector store database
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


# TODO: update docstring
def set_default_prompt_template() -> ChatPromptTemplate:
    """
    Create a Text QA Template for the query engine.

    Returns:
        ChatPromptTemplate: The initialized Text QA Template.
    """

    chat_text_msgs = [
        ChatMessage(
            role=MessageRole.USER,
            content=QUERY_PROMPT_TEMPLATE,
        ),
    ]

    return SYSTEM_PROMPT, ChatPromptTemplate(chat_text_msgs)


# TODO: move to cost_calculation.py, remove env variables, add llm_class_name input, update docstring
def initialize_token_counting(encoding_model: str = 'gpt-3.5-turbo'):
    """
    Initializes the Token Counting Handler for tracking token usage.

    Returns:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
        callback_manager (CallbackManager): Callback Manager with Token Counting Handler included.
    """
    enable_token_counting = read_env_variable('ENABLE_TOKEN_COUNTING',
                                              DEFAULT_ENABLE_TOKEN_COUNTING).upper() == 'TRUE'
    if not enable_token_counting:
        logger.info('Token Counting Handler is not enabled.')
        return None

    # Initialize the Token Counting Handler
    token_counter = generate_token_counter(encoding_model=encoding_model)

    # Initialize Callback Manager and add Token Counting Handler
    callback_manager = CallbackManager([token_counter])

    return token_counter, callback_manager


def generate_token_counter(encoding_model: str = 'gpt-3.5-turbo'):
    """
    Generates a Token Counting Handler for tracking token usage.

    Parameters:
        encoding_model (str): The name of the encoding model to use.

    Returns:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
    """
    # Initialize the Token Counting Handler
    tokenizer = tiktoken.encoding_for_model(encoding_model).encode  # Update the model name as needed
    token_counter = TokenCountingHandler(tokenizer=tokenizer, verbose=True)

    return token_counter


def calculate_total_cost(token_counter: TokenCountingHandler, model_name='gpt-3.5-turbo'):
    """
    Calculate the total cost based on the token usage and model.

    Parameters:
        token_counter (TokenCountingHandler): Token Counting Handler initialized with the tokenizer.
        model_name (str): The name of the model being used.

    Returns:
        float: The total cost in USD.
    """
    model_cost_info = MODEL_COST.get(model_name, {})
    if not model_cost_info:
        raise ValueError(f'Cost information for model {model_name} is not available.')

    prompt_token_count = token_counter.prompt_llm_token_count
    completion_token_count = token_counter.completion_llm_token_count

    prompt_cost = (prompt_token_count /
                   model_cost_info['prompt']['unit']) * model_cost_info['prompt']['price']
    completion_cost = (completion_token_count /
                       model_cost_info['completion']['unit']) * model_cost_info['completion']['price']

    total_cost = prompt_cost + completion_cost

    return total_cost


# TODO: Delete this function?
def log_total_cost(token_counter: TokenCountingHandler):
    """
    Logs the total cost based on token usage if ENABLE_TOKEN_COUNTING is set to True in the environment
    variables.

    Parameters:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
    """
    enable_token_counting = read_env_variable('ENABLE_TOKEN_COUNTING',
                                              DEFAULT_ENABLE_TOKEN_COUNTING).lower() == 'true'

    if enable_token_counting:
        total_cost = calculate_total_cost(token_counter)
        logger.info(f'Total cost for this query: ${total_cost} USD')
    else:
        logger.info('Token counting and cost logging are disabled.')
