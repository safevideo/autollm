from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.llms.base import LLM

from autollm.auto.query_engine import AutoQueryEngine


def test_auto_llm():
    query_engine = AutoQueryEngine.from_parameters()

    response = query_engine.query("What is the meaning of life?")

    # Check if the response is not None
    assert response.response is not None
