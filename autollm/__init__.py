"""AutoLLM: A Base Package for Large Language Model Applications.

This package provides automated integrations with leading large language models
and vector databases, along with various utility functions.
"""

__version__ = '0.0.13'
__author__ = 'safevideo'
__license__ = 'AGPL-3.0'

from autollm.auto.fastapi_app import AutoFastAPI
from autollm.auto.llm import AutoLLM
from autollm.auto.query_engine import AutoQueryEngine
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store_index import AutoVectorStoreIndex

__all__ = ['AutoLLM', 'AutoServiceContext', 'AutoVectorStoreIndex', 'AutoQueryEngine', 'AutoFastAPI']
