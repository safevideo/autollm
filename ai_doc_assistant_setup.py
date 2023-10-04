import logging

from typing import Tuple
from dotenv import load_dotenv
from os.path import dirname, join
import os

from llama_index import (
    VectorStoreIndex,
    PromptHelper,
    ServiceContext,
    set_global_service_context
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import get_response_synthesizer, BaseSynthesizer
from llama_index.text_splitter import TokenTextSplitter

from llama_utils import initialize_or_load_index


# Initialize logger
logger = logging.getLogger(__name__)

def read_env_variable(variable_name: str, default_value: str = None) -> str:
    """Reads an environment variable, returning a default value if not found."""
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    return os.getenv(variable_name, default_value)


max_tokens = int(read_env_variable("MAX_TOKENS", 1024))
chunk_size = int(read_env_variable("CHUNK_SIZE", 1024))
chunk_overlap = int(read_env_variable("CHUNK_OVERLAP", 20))
context_window = int(read_env_variable("CONTEXT_WINDOW", 4096))
similarity_top_k = int(read_env_variable("SIMILARITY_TOP_K", 4))

# Define the system prompt and query prompt template
system_prompt = '''
You are an AI document assistant specialized in specialized in retrieving and summarizing information from a database of documents.
Your purpose is to help users find the most relevant and accurate answers to their questions based on the documents you have access to.
You can answer questions based on the information available in the documents.
Your answers should be detailed, accurate, and directly related to the query.
Do not hallucinate or make things up. Stick to the facts in the documents.
'''

query_prompt_template = '''
The document information is below.
---------------------
{context_str}
---------------------
Using the document information and your capabilities, answer the following query: {query_str}
'''

# Text QA Prompt
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
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

# Initialize LLM based on the backend selection
llm = initialize_llm()

node_parser = SimpleNodeParser.from_defaults(
    text_splitter=TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
)
prompt_helper = PromptHelper(
    context_window=context_window,
    num_output=256,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    node_parser=node_parser,
    prompt_helper=prompt_helper,
)
set_global_service_context(service_context)

# Create a custom query engine class
class AIDocQueryEngine(CustomQueryEngine):
    """AI Document Assistant Query Engine."""

    retriever: BaseRetriever

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            query_prompt_template.format(context_str=context_str, query_str=query_str)
        )
        return str(response)

def initialize_query_engine(docs_path: str, persist_index: bool = True) -> Tuple[VectorStoreIndex, bool]:
    """
    Initialize or load the Vector Store Index and return a query engine.

    Parameters:
        docs_path (str): Path to the documents folder.
        persist_index (bool): Whether to persist the index to disk.

    Returns:
        query_engine: The initialized or loaded query engine.
    """

    # Initialize or load the vector store index
    index, _ = initialize_or_load_index(docs_path=docs_path, persist_index=persist_index)
    
    # Create a retriever from the index
    retriever = index.as_retriever()

    # Initialize the query engine
    query_engine = AIDocQueryEngine(retriever=retriever)

    logger.info("Query engine successfully initialized.")
    return query_engine
