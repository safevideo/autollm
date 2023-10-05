# Desc: Utility functions for llama index.
import logging

import pinecone
from llama_index import VectorStoreIndex, StorageContext, ServiceContext, PromptHelper, set_global_service_context
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.text_splitter import TokenTextSplitter

from pathlib import Path
from typing import List, Type, Union

from .env_utils import read_env_variable, validate_environment_variables
from .git_utils import clone_or_pull_repository
from .hash_utils import check_for_changes
from .markdown_processing import get_markdown_files, process_and_get_documents
from .multimarkdown_reader import MultiMarkdownReader

logger = logging.getLogger(__name__)


def update_index_for_changed_files(index: Type[VectorStoreIndex], files: List[str]):
    """
    Update the index with the changed markdown files.

    This function first deletes all the old documents associated with the changed files
    from the index and then inserts the updated documents.

    Args:
        index (Type[BaseIndex]): The LlamaIndex object to be updated.
        files (List[str]): List of markdown files that have changed.

    Returns:
        None
    """
    # Initialize a MultiMarkdownReader object
    markdown_reader = MultiMarkdownReader()

    # Loop through each file in the list of changed files
    for file in files:
        # Initialize an empty list to store existing doc_ids
        existing_doc_ids = []

        # Iterate over the items in index.ref_doc_info
        for key, value in index.ref_doc_info.items():
            # Check if 'original_file_path' in metadata matches the file path
            if value.metadata.get('original_file_path') == str(file):
                # Append the key (doc_id) to the existing_doc_ids list
                existing_doc_ids.append(key)

        # Delete old documents related to the current file from the index
        for doc_id in existing_doc_ids:
            index.delete_ref_doc(doc_id, delete_from_docstore=True)
 
        # Parse the updated file into a list of Header Documents
        new_documents = markdown_reader.load_data(file)

        # Insert the new documents into the index
        for doc in new_documents:
            index.insert(doc)


def process_and_update_docs(index: VectorStoreIndex, docs_path: Path):
    """Process markdown files and update the Vector Store Index.

    Parameters:
        index (VectorStoreIndex): The Vector Store Index.
        docs_path (Path): Base directory to search for markdown files.
    """
    # Get the list of all markdown files in the repository
    markdown_files = get_markdown_files(docs_path)

    # Identify files that have changed since the last update
    markdown_files_to_update = check_for_changes(markdown_files)

    # If it's not initial load and there are files to update
    if markdown_files_to_update:
        # Update the index with new documents
        update_index_for_changed_files(index, markdown_files_to_update)
    else:
        logger.info("No changes detected.")


def initialize_database(index_name: str, docs_path: Path, read_as_single_doc: bool) -> None:
    """
    Initialize the database with documents from the specified directory path.
    
    This function initializes the database by reading the document data from a
    given directory path and storing it in a vector database. The function
    uses Pinecone to manage the vector database.
    
    Parameters:
        index_name (str): The name of the Pinecone index_name to load.
        docs_path (Path): Base directory to search for markdown files.
        read_as_single_doc (bool): Flag to read entire markdown as a single document.
        
    Returns:
        None: This function returns None and is used for its side effects.
    """
    logger.info("Initializing the Pinecone database.")

    required_env_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    validate_environment_variables(required_env_vars)

    # Read environment variables for Pinecone initialization
    api_key = read_env_variable("PINECONE_API_KEY")
    environment = read_env_variable("PINECONE_ENVIRONMENT")

    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment=environment)
    # Dimensions are for text-embedding-ada-002
    pinecone.create_index(
        "quickstart",
        dimension=1536,
        metric="euclidean",
        pod_type="p1"
    )
    index = pinecone.Index(index_name)

    # Construct vector store
    vector_store = PineconeVectorStore(pinecone_index=index)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Process documents and load them into Pinecone Index
    documents = process_and_get_documents(docs_path, read_as_single_doc=read_as_single_doc)

    # create index, which will insert documents/vectors to pinecone
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return index


def update_database(
    git_repo_url: str, 
    git_repo_path: Path, 
    index_name: str,
    docs_path: Path
) -> None:
    """
    Updates the vector database by performing the following tasks:
    1. Clone or pull the latest repository containing markdown files.
    2. Connect to the the existing vector store database.
    3. Update the index with changed markdown files.

    Parameters:
        git_repo_url (str): URL of the git repository to clone or pull.
        git_repo_path (Path): Local path to clone the git repository.
        index_name (str): The name of the Pinecone index_name to load.
        docs_path (Path): Base directory to search for markdown files.

    Returns:
        None
    """
    logger.info("Starting to update the vector database.")

    # Step 1: Clone or pull the git repository to get the latest markdown files
    clone_or_pull_repository(git_repo_url, git_repo_path)

    # Step 2: Load the existing vector store index into memory
    connect_database(index_name=index_name)

    # Step 3: Update the index with changed markdown files
    process_and_update_docs(index=index_name, docs_path=docs_path)

    logger.info("Vector database successfully updated.")


def connect_database(index_name: str = "quickstart") -> Union[VectorStoreIndex, None]:
    """
    Conntect to existing database with data already loaded in.

    Parameters:
        index_name (str): The name of the Pinecone index to connect to. Default is 'quickstart'.

    Returns:
        VectorStoreIndex: The loaded vector store index.
        None: If the index could not be loaded.

    Raises:
        Exception: Detailed exception information if the index fails to load.
    """
    required_env_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    validate_environment_variables(required_env_vars)

    api_key = read_env_variable("PINECONE_API_KEY")
    environment = read_env_variable("PINECONE_ENVIRONMENT")

    try:
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Create an index instance that targets the given index_name
        pinecone_index = pinecone.Index(index_name)
        
        # Initialize an instance of the Pinecone vector store module
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # Connect to the vectore store with the data already loaded in
        loaded_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        logger.info(f"Successfully connected to the database with index: {index_name}")
        
        return loaded_index

    except Exception as e:
        logger.error(f"Failed to load Pinecone vector store index: {index_name}. Error: {e}")
        return None


# TODO: Move these to a config file?
# Define the system prompt and query prompt template
system_prompt = '''
You are an AI document assistant specialized in retrieving and summarizing information from a database of documents.
Your purpose is to help users find the most relevant and accurate answers to their questions based on the documents you have access to.
You can answer questions based on the information available in the documents.
Your answers should be detailed, accurate, and directly related to the query.
When answering the questions, mostly rely on the info in documents.
'''

query_prompt_template = '''
The document information is below.
---------------------
{context_str}
---------------------
Using the document information and not prior knowledge,
answer the query.
Query: {query_str}
Answer:
'''

# TODO: Add llm options
def initialize_service_context() -> ServiceContext:
    """
    Initialize and configure the service context utility container for LlamaIndex
    index and query classes.

    Returns:
        ServiceContext: The initialized service context.
    """
    chunk_size = int(read_env_variable("CHUNK_SIZE", 1024))
    chunk_overlap = int(read_env_variable("CHUNK_OVERLAP", 20))
    context_window = int(read_env_variable("CONTEXT_WINDOW", 4096))

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )
    prompt_helper = PromptHelper(
        context_window=context_window,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None
    )

    embed_model = OpenAIEmbedding() # text-embedding-ada-002 by default
    service_context = ServiceContext.from_defaults(
        prompt_helper=prompt_helper,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    set_global_service_context(service_context)
    logger.info("Service context for index and query has initialized successfully.")


def create_text_qa_template(system_prompt: str, query_prompt_template: str) -> ChatPromptTemplate:
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
