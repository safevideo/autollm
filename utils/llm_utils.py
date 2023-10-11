# Desc: Utility functions for llama index.
import logging
from typing import Union, Sequence, Optional
import tiktoken

from llama_index import (ServiceContext, set_global_service_context, Document)
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import Anyscale, OpenAI, PaLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole
from llama_index.text_splitter import TokenTextSplitter

from utils.constants import (
    DEFAULT_INDEX_NAME,
    DEFAULT_VECTORE_STORE_TYPE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_LLM_BACKEND,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL_NAME,
    DEFAULT_PALM_MODEL_NAME,
    DEFAULT_ANYSCALE_MODEL_NAME,
    DEFAULT_ENABLE_TOKEN_COUNTING,
    MODEL_COST,
)

from vectorstores.auto import AutoVectorStore

from .env_utils import read_env_variable
from .hash_utils import check_for_changes
from .templates import QUERY_PROMPT_TEMPLATE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def initialize_database(
    documents: Sequence[Document],
    vectore_store_type: str = DEFAULT_VECTORE_STORE_TYPE
) -> None:
    """
    Initializes the vector database for the first time from given documents.

    Parameters:
        documents (Sequence[Document]): List of documents to initialize the vector store with.
        vectore_store_type (str): Type of vector store to use ('qdrant', 'pinecone', etc.).

    Returns:
        None
    """
    logger.info("Initializing vector store")

    # Connect to the existing vector store database
    vector_store = AutoVectorStore.create(vector_store_type=vectore_store_type, collection_name=DEFAULT_INDEX_NAME)
    vector_store.initialize_vectorindex()
    vector_store.connect_vectorstore()
    
    logger.info("Updating vector store with documents")

    # Update the index with the documents
    vector_store.overwrite_vectorindex(documents)

    logger.info("Vector database successfully initialized.")


def update_database(
    documents: Sequence[Document],
    vectore_store_type: str
) -> None:
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
    logger.info("Updating vector store")

    # Get changed document ids using the hash of the documents available in the vector store index item metadata
    vector_store = AutoVectorStore.create(vector_store_type=vectore_store_type, index_name=DEFAULT_INDEX_NAME)
    changed_documents, deleted_document_ids = check_for_changes(documents, vector_store)

    # Update the index with the changed documents
    vector_store.update_vectorindex(changed_documents)
    vector_store.delete_documents_by_id(deleted_document_ids)

    logger.info("Vector database successfully updated.")


def initialize_service_context(
        llm_backend: str = DEFAULT_LLM_BACKEND,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        openai_model_name: Optional[str] = DEFAULT_OPENAI_MODEL_NAME,
        palm_model_name: Optional[str] = DEFAULT_PALM_MODEL_NAME,
        anyscale_model_name: Optional[str] = DEFAULT_ANYSCALE_MODEL_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: CallbackManager = None
) -> ServiceContext:
    """
    Initialize and configures the service context utility container for LlamaIndex
    index and query classes, setting as the global default.

    Parameters:
        chunk_size (int): Size of the chunks to split the text into, in number of tokens.
        chunk_overlap (int): Number of tokens to overlap between chunks.
        context_window (int): Number of tokens to include in the context window.
        callback_manager (CallbackManager): Callback Manager to be included in the service context.

    Returns:
        None
    """
    # Initialize LLM based on the backend selection, default is OpenAI
    llm = initialize_llm(
        llm_backend=llm_backend,
        max_tokens=max_tokens,
        openai_model_name=openai_model_name,
        palm_model_name=palm_model_name,
        anyscale_model_name=anyscale_model_name
    )

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )

    # TODO: add system prompt and query prompt template directly to service context instead of qa_template
    embed_model = OpenAIEmbedding() # text-embedding-ada-002 by default
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
        callback_manager=callback_manager,
        context_window=context_window,
        num_output=256
    )
    
    set_global_service_context(service_context)
    logger.info("Service context initialized successfully.")


def create_text_qa_template(
    system_prompt: str = SYSTEM_PROMPT,
    query_prompt_template: str = QUERY_PROMPT_TEMPLATE
) -> ChatPromptTemplate:
    """
    Create a Text QA Template for the query engine.

    Parameters:
        system_prompt (str): The system prompt string.
        query_prompt_template (str): The query prompt template string.

    Returns:
        ChatPromptTemplate: The initialized Text QA Template.
    """

    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=system_prompt,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=query_prompt_template,
        ),
    ]

    return ChatPromptTemplate(chat_text_qa_msgs)


def initialize_llm(
        llm_backend: str = DEFAULT_LLM_BACKEND,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        openai_model_name: Optional[str] = DEFAULT_OPENAI_MODEL_NAME,
        palm_model_name: Optional[str] = DEFAULT_PALM_MODEL_NAME,
        anyscale_model_name: Optional[str] = DEFAULT_ANYSCALE_MODEL_NAME
) -> Union[OpenAI, PaLM, Anyscale]:
    """
    Initializes the language model based on the backend selection, returning the initialized LLM object.
    
    Parameters:
        llm_backend (str): The LLM backend to use (e.g., 'OPENAI', 'PALM', 'ANYSCALE'). Defaults to OPENAI.
        max_tokens (int): The maximum number of tokens to use for the LLM.
        openai_model_name (Optional[str]): The name of the OpenAI model to use.
        palm_model_name (Optional[str]): The name of the PaLM model to use.
        anyscale_model_name (Optional[str]): The name of the Anyscale model to use.
    Returns:
        llm (Union[OpenAI, PaLM, Anyscale]): Initialized LLM object.
    """
    if llm_backend == "OPENAI":
        return OpenAI(temperature=0.1, model=openai_model_name, max_tokens=max_tokens)
    elif llm_backend == "PALM":
        return PaLM(model_name=palm_model_name, num_output=max_tokens)
    elif llm_backend == "ANYSCALE":
        return Anyscale(model=anyscale_model_name, max_tokens=max_tokens)
    else:
        raise ValueError(f"Invalid LLM_BACKEND: {llm_backend}")


# TODO: Don't read from environment variables.
def initialize_token_counting(encoding_model: str = "gpt-3.5-turbo"):
    """
    Initializes the Token Counting Handler for tracking token usage.
    
    Returns:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
        callback_manager (CallbackManager): Callback Manager with Token Counting Handler included.
    """
    enable_token_counting = read_env_variable("ENABLE_TOKEN_COUNTING", DEFAULT_ENABLE_TOKEN_COUNTING).upper() == "TRUE"
    if not enable_token_counting:
        logger.info("Token Counting Handler is not enabled.")
        return None
    
    # Initialize the Token Counting Handler
    token_counter = generate_token_counter(encoding_model=encoding_model)

    # Initialize Callback Manager and add Token Counting Handler
    callback_manager = CallbackManager([token_counter])
    
    return token_counter, callback_manager


def generate_token_counter(encoding_model: str = "gpt-3.5-turbo"):
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


def calculate_total_cost(token_counter, model_name="gpt-3.5-turbo"):
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
        raise ValueError(f"Cost information for model {model_name} is not available.")

    prompt_token_count = token_counter.prompt_llm_token_count
    completion_token_count = token_counter.completion_llm_token_count

    prompt_cost = (prompt_token_count / model_cost_info['prompt']['unit']) * model_cost_info['prompt']['price']
    completion_cost = (completion_token_count / model_cost_info['completion']['unit']) * model_cost_info['completion']['price']

    total_cost = prompt_cost + completion_cost

    return total_cost


# TODO: Delete this function.
def log_total_cost(token_counter):
    """
    Logs the total cost based on token usage if ENABLE_TOKEN_COUNTING is set to True in the environment variables.
    
    Parameters:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
    """
    enable_token_counting = read_env_variable("ENABLE_TOKEN_COUNTING", DEFAULT_ENABLE_TOKEN_COUNTING).lower() == "true"
    
    if enable_token_counting:
        total_cost = calculate_total_cost(token_counter)
        logger.info(f"Total cost for this query: ${total_cost} USD")
    else:
        logger.info("Token counting and cost logging are disabled.")