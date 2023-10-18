import logging

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from autollm.app.docs import description, openapi_url, tags_metadata, terms_of_service, title, version
from autollm.app.utils import load_config_and_initialize_engines

logging.basicConfig(level=logging.INFO)


class QueryPayload(BaseModel):
    task: str = Field(..., description="Task to execute")
    user_query: str = Field(..., description="User's query")


# Function to create the FastAPI web app
def create_web_app(config_file_path: str, env_file_path: str = None):
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        openapi_url=openapi_url,
        terms_of_service=terms_of_service,
        openapi_tags=tags_metadata,
    )

    query_engines = load_config_and_initialize_engines(config_file_path, env_file_path)

    @app.post("/query")
    async def query(payload: QueryPayload):
        task = payload.task
        user_query = payload.user_query

        if task not in query_engines:
            raise HTTPException(status_code=400, detail="Invalid task name")

        # Use the appropriate query engine for the task
        query_engine = query_engines[task]
        response = query_engine.query(user_query)

        return response

    return app


# For demonstration, let's assume we have a config.yaml with task configurations and an optional .env file
# This function call would typically be in your main application file
# app = create_web_app("path/to/config.yaml", "path/to/.env")
