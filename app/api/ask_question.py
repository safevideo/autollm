import logging
from fastapi import APIRouter

from utils import llm_utils

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/ask_question")
async def ask_question(user_query: str, index_name: str = "quickstart"):
    """
    Perform Text-Based Queries on Document Store
    
    This endpoint receives a natural language query from the user and returns the most relevant answer from the document store.

    Args:
        user_query (str): The natural language query from the user.
        index_name (str): The name of the vector store database index to connect to. Default is 'quickstart'.

    Returns:
        dict: The response containing the answer to the user's query.
    """
    # Initialize the service context
    service_context = llm_utils.initialize_service_context()

    # Initialize the query engine
    index = llm_utils.connect_database(index_name=index_name)
    query_engine = index.as_query_engine(service_context=service_context)

    # Query the engine
    response = query_engine.query(user_query)
    return response
