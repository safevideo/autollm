# Desc: Utility functions for llama index.
import logging
from pathlib import Path
from typing import List, Type, Union
import tiktoken

from llama_index import (PromptHelper, ServiceContext, VectorStoreIndex,
                         set_global_service_context)
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import Anyscale, OpenAI, PaLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole
from llama_index.storage.docstore.types import RefDocInfo
from llama_index.text_splitter import TokenTextSplitter

from utils.constants import PINECONE_INDEX_NAME
from vectorstores import \
    PineconeVS  # TODO: utilize vector store factory for generic use

from .env_utils import read_env_variable
from .git_utils import clone_or_pull_repository
from .hash_utils import check_for_changes
from .markdown_processing import process_and_get_documents
from .multimarkdown_reader import MultiMarkdownReader
from .templates import QUERY_PROMPT_TEMPLATE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def update_index_for_changed_files(index: Type[VectorStoreIndex], files: List[str], read_as_single_doc: bool = True):
    """
    Update the index with the changed markdown files.

    This function first deletes all the old documents associated with the changed files
    from the index and then inserts the updated documents.

    Args:
        index (Type[BaseIndex]): The LlamaIndex object to be updated.
        files (List[str]): List of markdown files that have changed.
        read_as_single_doc (bool): If True, read each markdown as a single document.

    Returns:
        None
    """
    # get all doc_id, filename pairs in the index

    filepath_to_doc_id = {}
    for doc_id, vector_object in index.ref_doc_info.items():
        vector_object: RefDocInfo
        filepath_to_doc_id[vector_object.metadata.get('original_file_path')] = doc_id

    # Loop through each file in the list of changed files
    for file in files:
        # Delete old documents related to the current file from the index
        if str(file) in filepath_to_doc_id:
            doc_id = filepath_to_doc_id[str(file)]
            index.delete_ref_doc(doc_id, delete_from_docstore=True)

    # Initialize a MultiMarkdownReader object
    updated_documents = MultiMarkdownReader(read_as_single_doc=read_as_single_doc).load_data_from_folder_or_files(files=files)

    # Insert the new documents into the index
    for updated_document in updated_documents:
        index.insert(updated_document)


def initialize_database(
        git_repo_url: str,
        git_repo_path: Path,
        read_as_single_doc: bool = True,
        relative_docs_path: Path = None
) -> None:
    """
    Initialize the database with documents from the specified directory path.

    This function initializes the database by reading the document data from a
    given directory path and storing it in a vector database. The function
    uses Pinecone to manage the vector database.

    Parameters:
        git_repo_url (str): URL of the git repository to clone or pull.
        git_repo_path (Path): Local path to clone the git repository.
        read_as_single_doc (bool): If True, read each markdown as a single document.
        relative_docs_path (Path): Relative path to the directory containing markdown files.

    Returns:
        None
    """

    logger.info("Getting repo files for initialize_database")

    # Step 1: Clone or pull the git repository to get the latest markdown files
    if relative_docs_path is None:
        relative_docs_path = Path("docs")  # TODO: read from utils/constants.py
    clone_or_pull_repository(git_repo_url, git_repo_path)
    docs_path = git_repo_path / relative_docs_path

    logger.info("Processing repo files to get documents")

    # Step 2: Process the markdown files and get the documents
    documents = process_and_get_documents(docs_path, read_as_single_doc=read_as_single_doc)

    logger.info("Initializing vector store")

    # Step 3: Connect to the existing vector store database
    pinecone_vs = PineconeVS(index_name=PINECONE_INDEX_NAME)  # TODO: utilize vector store factory for generic use
    pinecone_vs.initialize_vectorindex()

    logger.info("Updating vector store with documents")

    # Step 4: Update the index with the documents
    pinecone_vs.overwrite_vectorindex(documents)

    logger.info("Vector database successfully initialized.")


def update_database(
    git_repo_url: str,
    git_repo_path: Path,
    read_as_single_doc: bool = True,
    relative_docs_path: Path = None
) -> None:
    """
    Updates the vector database by performing the following tasks:
    1. Clone or pull the git repository to get the latest markdown files
    2. Process the markdown files and get the documents
    3. Get changed document ids using the hash of the documents available in the vector store index item metadata
    4. Update the index with the changed documents

    Parameters:
        git_repo_url (str): URL of the git repository to clone or pull.
        git_repo_path (Path): Local path to clone the git repository.
        read_as_single_doc (bool): If True, read each markdown as a single document.
        relative_docs_path (Path): Relative path to the directory containing markdown files.

    Returns:
        None
    """
    logger.info("Getting repo files for update_database")

    # Step 1: Clone or pull the git repository to get the latest markdown files
    if relative_docs_path is None:
        relative_docs_path = Path("docs")  # TODO: read from utils/constants.py
    clone_or_pull_repository(git_repo_url, git_repo_path)
    docs_path = git_repo_path / relative_docs_path

    logger.info("Processing repo files to get documents")

    # Step 2: Process the markdown files and get the documents
    documents = process_and_get_documents(docs_path, read_as_single_doc=read_as_single_doc)

    # Step 3: get changed document ids using the hash of the documents available in the vector store index item metadata
    pinecone_vs = PineconeVS(index_name=PINECONE_INDEX_NAME)  # TODO: utilize vector store factory for generic use
    changed_documents = check_for_changes(documents)

    # Step 4: Update the index with the changed documents
    pinecone_vs.update_vectorindex(changed_documents)


def initialize_service_context() -> ServiceContext:
    """
    Initialize and configure the service context utility container for LlamaIndex
    index and query classes.

    Returns:
        ServiceContext: The initialized service context.
    """
    chunk_size = int(read_env_variable("CHUNK_SIZE", 1024))  # TODO: read from utils/constants.py
    chunk_overlap = int(read_env_variable("CHUNK_OVERLAP", 20))  # TODO: read from utils/constants.py
    context_window = int(read_env_variable("CONTEXT_WINDOW", 4096))  # TODO: read from utils/constants.py

    # Initialize LLM based on the backend selection from environment variables
    llm = initialize_llm()  # Default is OpenAI

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )
    # TODO: PromptHelper deprecated, move parameters into ServiceContext
    prompt_helper = PromptHelper(
        context_window=context_window,
        num_output=256,
        chunk_overlap_ratio=0.1,
    )

    callback_manager = initialize_token_counting()
    # TODO: add system prompt and query prompt template directly to service context instead of qa_template
    embed_model = OpenAIEmbedding() # text-embedding-ada-002 by default
    service_context = ServiceContext.from_defaults(
        llm=llm,
        prompt_helper=prompt_helper,
        embed_model=embed_model,
        node_parser=node_parser,
        callback_manager=callback_manager
    )

    return service_context


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


def initialize_llm() -> Union[OpenAI, PaLM, Anyscale]:
    """Initializes the language model based on the backend selection, returning the initialized LLM object."""
    llm_backend = read_env_variable("LLM_BACKEND", "OPENAI")  # TODO: read from utils/constants.py
    max_tokens = int(read_env_variable("MAX_TOKENS", 1024))  # TODO: read from utils/constants.py

    openai_model_name = read_env_variable("OPENAI_MODEL_NAME", "gpt-3.5-turbo")  # TODO: read from utils/constants.py
    palm_model_name = read_env_variable("PALM_MODEL_NAME", "models/text-bison-001") # TODO: read from utils/constants.py
    anyscale_model_name = read_env_variable("ANYSCALE_MODEL_NAME", "meta-llama/Llama-2-70b-chat-hf")  # TODO: read from utils/constants.py

    if llm_backend == "OPENAI":
        return OpenAI(temperature=0.1, model=openai_model_name, max_tokens=max_tokens)
    elif llm_backend == "PALM":
        return PaLM(model_name=palm_model_name, num_output=max_tokens)
    elif llm_backend == "ANYSCALE":
        return Anyscale(model=anyscale_model_name, max_tokens=max_tokens)
    else:
        raise ValueError(f"Invalid LLM_BACKEND: {llm_backend}")


def initialize_token_counting():
    """
    Initializes the Token Counting Handler for tracking token usage.
    
    Returns:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
        callback_manager (CallbackManager): Callback Manager with Token Counting Handler included.
    """
    enable_token_counting = read_env_variable("ENABLE_TOKEN_COUNTING", "False").upper() == "TRUE"
    if not enable_token_counting:
        logger.info("Token Counting Handler is not enabled.")
        return None
    
    # Initialize the Token Counting Handler
    token_counter = generate_token_counter()

    # Initialize Callback Manager and add Token Counting Handler
    callback_manager = CallbackManager([token_counter])
    
    return token_counter, callback_manager


def generate_token_counter():
    """
    Generates a Token Counting Handler for tracking token usage.
    
    Returns:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
    """
    # Initialize the Token Counting Handler
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode  # Update the model name as needed
    token_counter = TokenCountingHandler(tokenizer=tokenizer, verbose=True)
    
    return token_counter


from .constants import MODEL_COST

def calculate_total_cost(token_counter, model_name="gpt-3.5-turbo"):
    """
    Calculate the total cost based on the token usage and model.

    Args:
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



def log_total_cost(token_counter):
    """
    Logs the total cost based on token usage if ENABLE_TOKEN_COUNTING is set to True in the environment variables.
    
    Args:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
    """
    enable_token_counting = read_env_variable("ENABLE_TOKEN_COUNTING", "True").lower() == "true"
    
    if enable_token_counting:
        total_cost = calculate_total_cost(token_counter)
        logger.info(f"Total cost for this query: ${total_cost} USD")
    else:
        logger.info("Token counting and cost logging are disabled.")