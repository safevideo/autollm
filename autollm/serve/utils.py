import logging
from typing import Dict

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autollm.auto.query_engine import AutoQueryEngine
from autollm.serve.docs import description, openapi_url, tags_metadata, terms_of_service, title, version

logging.basicConfig(level=logging.INFO)


# Function to load the configuration for tasks and initialize query engines
def load_config_and_initialize_engines(config_file_path: str,
                                       env_file_path: str = None) -> Dict[str, AutoQueryEngine]:
    # Optionally load environment variables from a .env file
    if env_file_path:
        load_dotenv(dotenv_path=env_file_path)

    # Load the YAML configuration file
    with open(config_file_path) as f:
        config = yaml.safe_load(f)

    # Initialize query engines based on the config
    query_engines = {}
    for task_params in config['tasks']:
        task_name = task_params.pop('name')
        query_engines[task_name] = AutoQueryEngine.from_parameters(**task_params)

    return query_engines


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
