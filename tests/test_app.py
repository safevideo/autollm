# Importing the necessary modules for testing and mocking
from unittest.mock import mock_open, patch

from fastapi.testclient import TestClient
from llama_index.indices.query.base import BaseQueryEngine

from autollm.app.main import create_web_app, load_config_and_initialize_engines

# Mock file content that mimics a YAML configuration file
mock_file_content = "task1: {system_prompt: 'prompt1'}"


# 1. Testing Configuration Loading and Initialization
@patch('builtins.open', new_callable=mock_open, read_data=mock_file_content)
@patch('yaml.safe_load')
def test_load_config_and_initialize_engines(mock_safe_load, mock_file):
    # Mock the yaml.safe_load to return a sample config
    sample_config = {
        'summarize': {
            'system_prompt': 'You are a friendly chatbot that can summarize documents.'
        },
        'qa': {
            'system_prompt': 'You are a friendly chatbot that can answer questions.'
        }
    }
    mock_safe_load.return_value = sample_config

    query_engines = load_config_and_initialize_engines('mock_path')

    # Assert that the open function was called with the correct file path
    mock_file.assert_any_call('mock_path')

    # Validate the type and configuration of each query engine
    for task in sample_config.keys():
        assert isinstance(query_engines[task], BaseQueryEngine)


# 2. Testing FastAPI App Creation
@patch('builtins.open', new_callable=mock_open, read_data=mock_file_content)
def test_create_web_app(mock_file):
    # Assuming the function create_web_app is being imported correctly
    app = create_web_app('mock_path')

    # Validate presence of the query endpoint
    assert any(route.path == "/query" for route in app.routes)


# 3. Testing Query Endpoint
@patch('builtins.open', new_callable=mock_open, read_data=mock_file_content)
def test_query_endpoint(mock_file):
    # Assuming the function create_web_app is being imported correctly
    app = create_web_app('mock_path')
    client = TestClient(app)

    # Test with a valid task
    response = client.post("/query", json={"task": "task1", "user_query": "test query"})
    assert response.status_code == 200

    # Test with an invalid task
    response = client.post("/query", json={"task": "invalid_task", "user_query": "test query"})
    assert response.status_code == 400
