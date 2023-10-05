from dotenv import load_dotenv
from os.path import dirname, join
import os

# Load environment variables from .env file
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


def read_env_variable(variable_name: str, default_value: str = None) -> str:
    """Reads an environment variable, returning a default value if not found."""
    return os.getenv(variable_name, default_value)


def validate_environment_variables(required_vars: list) -> None:
    """Validates that all required environment variables are set, raising an error if any of them is not set."""
    for var in required_vars:
        if var not in os.environ:
            raise EnvironmentError(f"Environment variable {var} is not set")
