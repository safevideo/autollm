from llama_index import Document
from llama_index.query_engine import BaseQueryEngine

from autollm.auto.query_engine import AutoQueryEngine


def test_auto_query_engine_from_parameters():
    documents = [Document.example()]
    vector_store_params = {"vector_store_type": "SimpleVectorStore"}
    query_engine = AutoQueryEngine.from_parameters(
        documents=documents, vector_store_params=vector_store_params)

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)
