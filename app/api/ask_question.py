import logging

from fastapi import APIRouter

from utils import llm_utils
from utils.constants import DEFAULT_INDEX_NAME, DEFAULT_VECTORE_STORE_TYPE

from vectorstores import auto

router = APIRouter()
logger = logging.getLogger(__name__)

token_counter, callback_manager = llm_utils.initialize_token_counting()

# Initialize the service context
llm_utils.initialize_service_context(callback_manager=callback_manager)

# Create the text QA template for the query engine
text_qa_template = llm_utils.create_text_qa_template()

# Initialize the query engine
vector_store = auto.AutoVectorStore().create(DEFAULT_VECTORE_STORE_TYPE, index_name=DEFAULT_INDEX_NAME)
vector_store.connect_vectorstore()
query_engine = vector_store.vectorindex.as_query_engine(text_qa_template=text_qa_template)


@router.get("/ask_question")
async def ask_question(user_query: str):
    """
    Perform Text-Based Queries on Document Store
    
    This endpoint receives a natural language query from the user and returns the most relevant answer from the document store.

    Args:
        user_query (str): The natural language query from the user.

    Returns:
        response (str): The response containing the answer to the user's query.
    """
    # Query the engine
    response = query_engine.query(user_query)
    llm_utils.log_total_cost(token_counter=token_counter)
    token_counter.reset_counts()
    return response.response    # extracts the response text
