from llama_index import Document

from autollm.auto.query_engine import AutoQueryEngine


def test_auto_query_engine():
    documents = Document.example()
    vector_store_params = {"vector_store_type": "VectorStoreIndex", "documents": documents}
    query_engine = AutoQueryEngine.from_parameters(vector_store_params=vector_store_params)

    response = query_engine.query("What is the meaning of life?")

    # Check if the response is not None
    assert response.response is not None
