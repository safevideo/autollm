from llama_index import Document, VectorStoreIndex
from llama_index.query_engine import BaseQueryEngine

from autollm.auto.query_engine import AutoQueryEngine

documents = [Document.example()]


def test_auto_query_engine_from_defaults():
    vector_store_type = "SimpleVectorStore"
    query_engine = AutoQueryEngine.from_defaults(documents=documents, vector_store_type=vector_store_type)

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)


def test_auto_query_engine_from_instances():
    vector_store_index = VectorStoreIndex.from_documents(documents=documents)

    query_engine = AutoQueryEngine.from_instances(vector_store_index=vector_store_index, service_context=None)

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)


def test_auto_query_engine_from_config():
    query_engine = AutoQueryEngine.from_config(config_file_path="tests/config.yaml", documents=documents)

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)
