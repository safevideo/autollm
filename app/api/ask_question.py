import logging
from fastapi import APIRouter

from utils import llm_utils

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the service context
llm_utils.initialize_service_context()

# Create the text QA template for the query engine
text_qa_template = llm_utils.create_text_qa_template()

# Initialize the query engine
index = llm_utils.connect_database()
query_engine = index.as_query_engine(text_qa_template=text_qa_template)


@router.get("/ask_question")
async def ask_question(user_query: str):
    """
    Perform Text-Based Queries on Document Store
    
    This endpoint receives a natural language query from the user and returns the most relevant answer from the document store.

    Args:
        user_query (str): The natural language query from the user.

    Returns:
        dict: The response containing the answer to the user's query.
    """
    # Query the engine
    response = query_engine.query(user_query)
    return response
