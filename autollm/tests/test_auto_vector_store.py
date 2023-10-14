from pathlib import Path

from llama_index import Document, ServiceContext

from autollm.auto.vector_store import AutoVectorStore


def test_auto_vector_store():
    vector_store = AutoVectorStore.from_defaults(vector_store_type="in_memory")

    vector_store.initialize_vectorindex()
    vector_store.connect_vectorstore()

    query_engine = vector_store.vectorstore.as_query_engine()

    response = query_engine.query("What is the meaning of life?")

    # Check if the response is not None
    assert response.response is not None
