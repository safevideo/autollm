# Purpose: FastAPI documentation configuration.
# Metadata
title = "AutoLLM Query Engine"
description = """
This is a FastAPI service for AutoLLM's natural language query engine.
It's designed to query multiple and big documents and get the most relevant results.
"""
version = "0.0.1"
openapi_url = "/api/v1/openapi.json"
terms_of_service = "Local Deployment, All Rights Reserved."
tags_metadata = [
    {
        "name": "query",
        "description": "Operations related to querying the header-documents."
    },
]
