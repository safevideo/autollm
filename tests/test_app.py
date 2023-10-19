# Importing the necessary modules for testing and mocking

from fastapi.testclient import TestClient
from llama_index.indices.query.base import BaseQueryEngine

from autollm.serve.utils import create_web_app, load_config_and_initialize_engines


def test_load_config_and_initialize_engines():
    # Mock the yaml.safe_load to return a sample config
    sample_config = {
        'summarize': {
            'system_prompt': 'You are a friendly chatbot that can summarize documents.'
        },
        'qa': {
            'system_prompt': 'You are a friendly chatbot that can answer questions.'
        }
    }

    query_engines = load_config_and_initialize_engines('tests/config.yaml')

    # Validate the type and configuration of each query engine
    for task in sample_config.keys():
        assert isinstance(query_engines[task], BaseQueryEngine)


# 2. Testing FastAPI App Creation
def test_create_web_app():
    # Assuming the function create_web_app is being imported correctly
    app = create_web_app('tests/config.yaml')

    # Validate presence of the query endpoint
    assert any(route.path == "/query" for route in app.routes)


# 3. Testing Query Endpoint
def test_query_endpoint():
    # Create the FastAPI app with test configuration
    app = create_web_app('tests/config.yaml')
    client = TestClient(app)

    # Test with a valid task (changed "task1" to "summarize")
    response = client.post("/query", json={"task": "summarize", "user_query": "test query"})
    assert response.status_code == 200

    # Test with an invalid task
    response = client.post("/query", json={"task": "invalid_task", "user_query": "test query"})
    assert response.status_code == 400
