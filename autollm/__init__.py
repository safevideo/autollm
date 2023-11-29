"""AutoLLM: A Base Package for Large Language Model Applications.

This package provides automated integrations with leading large language models
and vector databases, along with various utility functions.
"""

__version__ = '0.1.2'
__author__ = 'safevideo'
__license__ = 'AGPL-3.0'

from autollm.auto.fastapi_app import AutoFastAPI
from autollm.auto.llm import AutoLiteLLM
from autollm.auto.query_engine import AutoQueryEngine
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store_index import AutoVectorStoreIndex
from autollm.utils.document_reading import (
    read_files_as_documents,
    read_github_repo_as_documents,
    read_webpage_as_documents,
    read_website_as_documents,
)

__all__ = [
    'AutoLiteLLM', 'AutoServiceContext', 'AutoVectorStoreIndex', 'AutoQueryEngine', 'AutoFastAPI',
    'read_files_as_documents', 'read_github_repo_as_documents', 'read_webpage_as_documents',
    'read_website_as_documents'
]
