# First, let's import the necessary modules and utility functions
from typing import Tuple
from llama_index import (
    VectorStoreIndex,
    PromptTemplate,
    ServiceContext,
    set_global_service_context
)
from llama_index.query_engine import CustomQueryEngine
from llama_index.retrievers import BaseRetriever
from llama_index.response_synthesizers import get_response_synthesizer, BaseSynthesizer
from llama_utils import initialize_or_load_index
import logging

# Initialize logger
logger = logging.getLogger(__name__)

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

    # Configure the global service context
    service_context = ServiceContext.from_defaults(
        query_engine=query_engine
    )
    set_global_service_context(service_context)

    logger.info("Query engine successfully initialized.")
    return query_engine
