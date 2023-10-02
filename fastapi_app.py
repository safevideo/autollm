from fastapi import FastAPI
from fastapi.responses import JSONResponse
from llama_utils import initialize_or_load_index
from pathlib import Path
import logging

# Metadata
title = "SafeVideo Query Engine"
description = """
This is a FastAPI service for SafeVideo's natural language query engine. 
It's designed to query Markdown documents and return responses based on a VectorStoreIndex.
"""
version = "1.0.0"
openapi_url = "/api/v1/openapi.json"
terms_of_service = "Local Deployment, All Rights Reserved."
tags_metadata = [
    {
        "name": "query",
        "description": "Operations related to querying the text data."
    },
    {
        "name": "health",
        "description": "Health check operations."
    },
]

# Initialize FastAPI and Logging
app = FastAPI(
    title=title,
    description=description,
    version=version,
    openapi_url=openapi_url,
    terms_of_service=terms_of_service,
    openapi_tags=tags_metadata,
)

logging.basicConfig(level=logging.INFO)

# Initialize or load the vector store index
folder_path = Path('./README.md')
index, initial_load = initialize_or_load_index(docs_path=folder_path)
query_engine = index.as_query_engine()

@app.get("/query/", tags=["query"])
async def read_query(user_query: str):
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

@app.get("/health/", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}
