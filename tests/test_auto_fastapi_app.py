# Importing the necessary modules for testing and mocking

from fastapi.testclient import TestClient
from llama_index import Document
from llama_index.indices.query.base import BaseQueryEngine

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
    from llama_index import VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents=documents)
    query_engine = index.as_query_engine()

    # Create the FastAPI app from the query engine
    app = AutoFastAPI.from_query_engine(query_engine=query_engine)

    # Validate presence of the query endpoint
    assert any(route.path == "/query" for route in app.routes)


def test_query_endpoint():
    # Create the FastAPI app with test configuration
    app = AutoFastAPI.from_config('tests/config.yaml', documents=documents)
    client = TestClient(app)

    # Test with a valid task (changed "task1" to "summarize")
    response = client.post("/query", json={"task": "summarize", "user_query": "test query"})
    assert response.status_code == 200

    # Test with an invalid task
    response = client.post("/query", json={"task": "invalid_task", "user_query": "test query"})
    assert response.status_code == 400
