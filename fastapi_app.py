from fastapi import FastAPI
from pathlib import Path
import logging

from ai_doc_assistant_setup import initialize_query_engine, initialize_service_context
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

# Initialize the service context
service_context = initialize_service_context()

# Initialize the query engine
query_engine = initialize_query_engine(docs_path=folder_path)

@app.get("/ask_question", tags=["ask"])
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

@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}
