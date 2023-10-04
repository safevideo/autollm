from dotenv import load_dotenv
from os.path import dirname, join
import os

def read_env_variable(variable_name: str, default_value: str = None) -> str:
    """Reads an environment variable, returning a default value if not found."""
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    return os.getenv(variable_name, default_value)