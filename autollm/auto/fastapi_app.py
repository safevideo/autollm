from typing import Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_index import Document
from llama_index.indices.query.base import BaseQueryEngine
from pydantic import BaseModel, Field

from autollm.serve.docs import description, openapi_url, tags_metadata, terms_of_service, title, version
from autollm.serve.utils import load_config_and_initialize_engines, stream_text_data


class FromConfigQueryPayload(BaseModel):
    task: str = Field(..., description="Task to execute")
    user_query: str = Field(..., description="User's query")
    streaming: Optional[bool] = Field(False, description="Flag to enable streaming of response")


class FromEngineQueryPayload(BaseModel):
    user_query: str = Field(..., description="User's query")
    streaming: Optional[bool] = Field(False, description="Flag to enable streaming of response")


class AutoFastAPI:
    """Creates an FastAPI instance from config.yaml or Llama-Index query engine."""

    @staticmethod
    def from_config(
            config_file_path: Optional[str] = None,
            env_file_path: Optional[str] = None,
            task_name_to_query_engine: Optional[dict] = None,
            documents: Optional[Sequence[Document]] = None,
            api_title: str = None,
            api_description: str = None,
            api_version: str = None,
            api_term_of_service: str = None) -> FastAPI:
        """
        Create an FastAPI instance from config.yaml and optionally a .env file. The app has a single endpoint
        /query that takes a QueryPayload and returns a QueryResponse.

        ```python
        from autollm.auto.fastapi_app import QueryPayload, AutoFastAPI
        import uvicorn
        import requests

        # Start the server using from_config class method
        app = AutoFastAPI.from_config(config.yaml", ".env")
        uvicorn.run(app, host="0.0.0.0", port=8000)

        # Post request to the server
        data = {
          "task": "task1",
          "user_query": "why so serious?"
        }
        response = requests.post('http://0.0.0.0:8000/query', json=data)
        print(response.json())
        ```

        Parameters:
            config_file_path (str): Path to the YAML configuration file.
            env_file_path (str): Path to the .env file.
            task_name_to_query_engine (dict): Dictionary mapping task names to query engines.
            documents (Sequence[Document]): Sequence of llama_index.Document instances.
            api_title (str): Title of the API.
            api_description (str): Description of the API.
            api_version (str): Version of the API.
            api_term_of_service (str): Term of service of the API.

        Returns:
            FastAPI: The initialized FastAPI instance.
        """

        if task_name_to_query_engine is None and config_file_path is None:
            raise ValueError("Either config_file_path or task_name_to_query_engine must be provided")

        if task_name_to_query_engine is not None and config_file_path is not None:
            raise ValueError("Only one of config_file_path or task_name_to_query_engine must be provided")

        if task_name_to_query_engine is not None and not isinstance(task_name_to_query_engine, dict):
            raise ValueError("task_name_to_query_engine must be a dictionary")

        app = FastAPI(
            title=title if api_title is None else api_title,
            description=description if api_description is None else api_description,
            version=version if api_version is None else api_version,
            openapi_url=openapi_url,
            terms_of_service=terms_of_service if api_term_of_service is None else api_term_of_service,
            openapi_tags=tags_metadata,
        )

        if config_file_path is not None:
            task_name_to_query_engine = load_config_and_initialize_engines(
                config_file_path, env_file_path, documents)

        @app.post("/query")
        async def query(payload: FromConfigQueryPayload):
            task = payload.task
            user_query = payload.user_query

            if task not in task_name_to_query_engine:
                raise HTTPException(status_code=400, detail="Invalid task name")

            # Use the appropriate query engine for the task
            query_engine: BaseQueryEngine = task_name_to_query_engine[task]
            response = query_engine.query(user_query)

            # Check if the response should be streamed
            if payload.streaming:
                return StreamingResponse(stream_text_data(response.response))

            return response.response

        return app

    @staticmethod
    def from_query_engine(
            query_engine: BaseQueryEngine,
            api_title: str = None,
            api_description: str = None,
            api_version: str = None,
            api_term_of_service: str = None) -> FastAPI:
        """
        Create an FastAPI instance from a llama-index query engine.

        ```python
        from autollm.auto.fastapi_app import QueryPayload, AutoFastAPI
        import uvicorn
        import requests

        # Start the server using from_query_engine class method
        app = AutoFastAPI.from_query_engine(query_engine)
        uvicorn.run(app, host="0.0.0.0", port=8000)

        # Post request to the server
        data = {
           "user_query": "why so serious?"
        }
        response = requests.post('http://0.0.0.0:8000/query', json=data)

        print(response.json())
        ```

        Parameters:
            query_engine (BaseQueryEngine): Query engine.
            api_title (str): Title of the API.
            api_description (str): Description of the API.
            api_version (str): Version of the API.
            api_term_of_service (str): Term of service of the API.

        Returns:
            FastAPI: The initialized FastAPI instance.
        """

        if not isinstance(query_engine, BaseQueryEngine):
            raise ValueError("query_engine must be a llama_index query engine")

        app = FastAPI(
            title=title if api_title is None else api_title,
            description=description if api_description is None else api_description,
            version=version if api_version is None else api_version,
            openapi_url=openapi_url,
            terms_of_service=terms_of_service if api_term_of_service is None else api_term_of_service,
            openapi_tags=tags_metadata,
        )

        @app.post("/query")
        async def query(payload: FromEngineQueryPayload):
            user_query = payload.user_query

            response = query_engine.query(user_query)

            # Check if the response should be streamed
            if payload.streaming:
                return StreamingResponse(stream_text_data(response.response))

            return response.response

        return app
