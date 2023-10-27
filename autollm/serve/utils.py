import logging
from typing import Dict, Optional, Sequence

import yaml
from dotenv import load_dotenv
from llama_index import Document
from llama_index.indices.query.base import BaseQueryEngine

from autollm.auto.query_engine import AutoQueryEngine
from autollm.serve.docs import description, openapi_url, tags_metadata, terms_of_service, title, version

logging.basicConfig(level=logging.INFO)


def load_config_and_dotenv(config_file_path: str, env_file_path: str = None) -> dict:
    """
    Load the YAML configuration file and optionally load environment variables from a .env file.

    Parameters:
        config_file_path (str): Path to the YAML configuration file.
        env_file_path (str): Path to the .env file.

    Returns:
        dict: The configuration dictionary.
    """
    # Optionally load environment variables from a .env file
    if env_file_path:
        load_dotenv(dotenv_path=env_file_path)

    # Load the YAML configuration file
    with open(config_file_path) as f:
        config = yaml.safe_load(f)

    return config


def load_config_and_initialize_engines(
        config_file_path: str,
        env_file_path: str = None,
        documents: Optional[Sequence[Document]] = None) -> Dict[str, BaseQueryEngine]:
    """
    Load the YAML configuration file and optionally load environment variables from a .env file. Initialize
    query engines based on the config.

    Parameters:
        config_file_path (str): Path to the YAML configuration file.
        env_file_path (str): Path to the .env file.
        documents (Sequence[Document]): Sequence of llama_index.Document instances.

    Returns:
        dict: The configuration mapping (task name -> query engine))
    """

    config = load_config_and_dotenv(config_file_path, env_file_path)

    # Initialize query engines based on the config
    query_engines = {}
    for task_params in config['tasks']:
        task_name = task_params.pop('name')
        task_params['vector_store_params']['documents'] = documents
        query_engines[task_name] = AutoQueryEngine.from_parameters(**task_params)

    return query_engines
