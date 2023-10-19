"""AutoLLM: A Base Package for Large Language Model Applications.

This package provides automated integrations with leading large language models
and vector databases, along with various utility functions.
"""

__version__ = '0.0.2'
__author__ = 'safevideo'
__license__ = 'AGPL-3.0'

from autollm.auto.llm import AutoLLM
from autollm.auto.query_engine import AutoQueryEngine
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store_index import AutoVectorStoreIndex
from autollm.serve.utils import create_web_app

__all__ = ['AutoLLM', 'AutoServiceContext', 'AutoVectorStoreIndex', 'AutoQueryEngine', 'create_web_app']
