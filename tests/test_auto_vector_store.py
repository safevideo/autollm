from llama_index import Document, VectorStoreIndex

from autollm.auto.vector_store_index import AutoVectorStoreIndex


def test_auto_vector_store():
    documents = Document.example()

    vector_store = AutoVectorStoreIndex.from_defaults(
        vector_store_type="VectorStoreIndex", documents=documents)

    # Check if the vector_store is an instance of VectorStoreIndex
    assert isinstance(vector_store, VectorStoreIndex)

    query_engine = vector_store.as_query_engine()

    response = query_engine.query("What is the meaning of life?")

    # Check if the response is not None
    assert response.response is not None
