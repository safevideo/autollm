import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def find_dotenv_file(start_path: Path) -> Path:
    """Searches for the .env file from start_path moving upwards."""
    current_path = start_path
    while current_path != Path('/'):
        dotenv_path = current_path / '.env'
        if dotenv_path.exists():
            return dotenv_path
        current_path = current_path.parent
    raise OSError('Could not find a .env file.')


# Find and load .env file
# dotenv_path = find_dotenv_file(Path(__file__).parent)
# load_dotenv(dotenv_path)


def read_env_variable(variable_name: str, default_value: str = None) -> str:
    """Reads an environment variable, returning a default value if not found."""
    return os.getenv(variable_name, default_value)


def validate_environment_variables(required_vars: list) -> None:
    """Validates that all required environment variables are set, raising an error if any of them is not
    set.
    """
    for var in required_vars:
        if var not in os.environ:
            raise OSError(f'Environment variable {var} is not set')


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
