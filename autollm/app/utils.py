from typing import Dict

import yaml
from dotenv import load_dotenv

from autollm.auto.query_engine import AutoQueryEngine


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
    for task, params in config.items():
        query_engines[task] = AutoQueryEngine.from_parameters(**params)

    return query_engines
