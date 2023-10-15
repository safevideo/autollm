# Purpose: FastAPI documentation configuration.
# Metadata
title = "SafeVideo Query Engine"
description = """
This is a FastAPI service for SafeVideo's natural language query engine.
It's designed to query multiple and big Markdown documents and get the most relevant results.
"""
version = "1.0.0"
openapi_url = "/api/v1/openapi.json"
terms_of_service = "Local Deployment, All Rights Reserved."
tags_metadata = [
    {
        "name": "ask",
        "description": "Operations related to querying the header-documents."
    },
    {
        "name": "health",
        "description": "Health check operations."
    },
]
