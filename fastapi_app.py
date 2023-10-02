from fastapi import FastAPI
from pathlib import Path
import logging

from llama_utils import initialize_or_load_index
from fastapi_docs import (
    title,
    description,
    version,
    openapi_url,
    terms_of_service,
    tags_metadata,
)

logging.basicConfig(level=logging.INFO)

# Initialize FastAPI and Logging
app = FastAPI(
    title=title,
    description=description,
    version=version,
    openapi_url=openapi_url,
    terms_of_service=terms_of_service,
    openapi_tags=tags_metadata,
)

# Initialize or load the vector store index
folder_path = Path("llama_index/docs")
index, initial_load = initialize_or_load_index(docs_path=folder_path)
query_engine = index.as_query_engine()

@app.get("/ask", tags=["ask"])
async def ask_docs(user_query: str):
    """
    Endpoint to perform text-based natural language queries.

    Args:
        user_query (str): The user's query.

    Returns:
        RESPONSE_TYPE: The query response.
    """
    # Query the engine
    response = query_engine.query(user_query)
    return response

@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}
