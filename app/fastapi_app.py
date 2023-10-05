import logging

from fastapi import FastAPI, HTTPException
from fastapi.logger import logger

from ai_doc_assistant_setup import initialize_service_context
from llama_utils import connect_database
from fastapi_docs import (
    title,
    description,
    version,
    openapi_url,
    terms_of_service,
    tags_metadata,
)

logging.basicConfig(handlers=[logging.StreamHandler()],
                    level=logging.INFO)

# Initialize FastAPI and Logging
app = FastAPI(
    title=title,
    description=description,
    version=version,
    openapi_url=openapi_url,
    terms_of_service=terms_of_service,
    openapi_tags=tags_metadata,
)

# Initialize the service context
service_context = initialize_service_context()

# Initialize the query engine
try:
    index = connect_database()
    query_engine = index.as_query_engine(service_context=service_context)
except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    raise HTTPException(status_code=500, detail="Database connection failed")

@app.get("/ask_question", tags=["ask"])
async def ask_question(user_query: str):
    """
    Performs text-based queries on the document store.

    This endpoint receives a natural language query from the user and fetches the most relevant answer from the document store.

    Parameters:
        user_query (str): The natural language query from the user.

    Returns:
        dict: The response containing the most relevant answer to the user's query.
    """
    try:
        # Execute the query on the engine
        response = query_engine.query(user_query)
        return response
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail="Query execution failed")


@app.get("/health", tags=["health"])
async def health_check():
    """
    Checks the health of the service and its dependencies.
    
    Returns:
        dict: The health status of the service.
    """
    try:
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy"}
