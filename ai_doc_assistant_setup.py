import logging

from llama_index import PromptHelper, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.prompts import ChatPromptTemplate, ChatMessage, MessageRole
from llama_index.text_splitter import TokenTextSplitter

from env_utils import read_env_variable


# Initialize logger
logger = logging.getLogger(__name__)

# Define the system prompt and query prompt template
system_prompt = '''
You are an AI document assistant specialized in specialized in retrieving and summarizing information from a database of documents.
Your purpose is to help users find the most relevant and accurate answers to their questions based on the documents you have access to.
You can answer questions based on the information available in the documents.
Your answers should be detailed, accurate, and directly related to the query.
Always answer the query using the provided context information
and not prior knowledge.
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

    service_context = ServiceContext.from_defaults(
        node_parser=node_parser,
        prompt_helper=prompt_helper,
    )

    return service_context


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
