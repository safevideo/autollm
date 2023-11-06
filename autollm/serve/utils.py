import logging
from typing import Dict, Optional, Sequence

from llama_index import Document
from llama_index.indices.query.base import BaseQueryEngine

from autollm.auto.query_engine import AutoQueryEngine
from autollm.utils.env_utils import load_config_and_dotenv

logging.basicConfig(level=logging.INFO)

STREAMING_CHUNK_SIZE = 16


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
        query_engines[task_name] = AutoQueryEngine.from_defaults(documents=documents, **task_params)

    return query_engines


def stream_text_data(text_data: str, chunk_size: int = STREAMING_CHUNK_SIZE):
    start = 0
    end = chunk_size
    while start < len(text_data):
        yield text_data[start:end]
        start = end
        end += chunk_size
