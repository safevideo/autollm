from llama_index import Document, VectorStoreIndex
from llama_index.query_engine import BaseQueryEngine

from autollm.auto.vector_store_index import AutoVectorStoreIndex


def test_auto_vector_store():
    documents = [Document.example()]

    vector_store_index = AutoVectorStoreIndex.from_defaults(
        vector_store_type="SimpleVectorStore", documents=documents)

    # Check if the vector_store_index is an instance of VectorStoreIndex
    assert isinstance(vector_store_index, VectorStoreIndex)

    query_engine = vector_store_index.as_query_engine()

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)
