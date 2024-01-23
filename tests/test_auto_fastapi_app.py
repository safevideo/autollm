# Importing the necessary modules for testing and mocking
import os

from fastapi.testclient import TestClient
from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llms import AzureOpenAI

from autollm.auto.fastapi_app import AutoFastAPI
from autollm.serve.utils import load_config_and_initialize_engines

# Mock the documents
documents = [Document.example()]

# Mock the yaml.safe_load to return a sample config
sample_config = {
    'summarize': {
        'system_prompt': 'You are a friendly chatbot that can summarize documents.'
    },
    'qa': {
        'system_prompt': 'You are a friendly chatbot that can answer questions.'
    }
}

# set the environment variables
azure_api_key = os.environ.get("AZURE_API_KEY")
azure_endpoint = os.environ.get("AZURE_API_BASE")
azure_api_version = os.environ.get("AZURE_API_VERSION")


def test_load_config_and_initialize_engines():
    # Mock the load_config_and_initialize_engines function with the sample config and documents
    query_engines = load_config_and_initialize_engines('tests/config.yaml', documents=documents)

    # Validate the type and configuration of each query engine
    for task in sample_config.keys():
        assert isinstance(query_engines[task], BaseQueryEngine)


def test_auto_fastapi_from_config():
    app = AutoFastAPI.from_config(config_file_path='tests/config.yaml', documents=documents)

    # Validate presence of the query endpoint
    assert any(route.path == "/query" for route in app.routes)


def test_auto_fastapi_from_query_engine():
    llm = AzureOpenAI(
        engine="gpt-35-turbo-1106",
        model="gpt-35-turbo-16k",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="text-embedding-ada-002",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context)
    query_engine = index.as_query_engine()

    # Create the FastAPI app from the query engine
    app = AutoFastAPI.from_query_engine(query_engine=query_engine)

    # Validate presence of the query endpoint
    assert any(route.path == "/query" for route in app.routes)


def test_query_endpoint_from_config():
    # Create the FastAPI app with test configuration
    app = AutoFastAPI.from_config('tests/config.yaml', documents=documents)
    client = TestClient(app)

    # Test with a valid task (changed "task1" to "summarize")
    response = client.post("/query", json={"task": "summarize", "user_query": "test query"})
    assert response.status_code == 200

    # Test with an invalid task
    response = client.post("/query", json={"task": "invalid_task", "user_query": "test query"})
    assert response.status_code == 400


def test_query_endpoint_from_query_engine():
    # Create llama-index query engine
    llm = AzureOpenAI(
        engine="gpt-35-turbo-1106",
        model="gpt-35-turbo-16k",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="text-embedding-ada-002",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context)
    query_engine = index.as_query_engine()

    # Create the FastAPI app from the query engine
    app = AutoFastAPI.from_query_engine(query_engine=query_engine)
    client = TestClient(app)

    # Test with a user query
    response = client.post("/query", json={"user_query": "why so serious?"})
    assert response.status_code == 200
