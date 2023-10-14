from llama_index import Document, VectorStoreIndex

from autollm.auto.service_context import AutoServiceContext


def test_auto_service_context():
    document = Document.example()

    service_context = AutoServiceContext.from_defaults()

    index = VectorStoreIndex.from_documents(documents=[document], service_context=service_context)

    query_engine = index.as_query_engine()

    response = query_engine.query("What is the meaning of life?")

    # Check if the response is not None
    assert response.response is not None

    # Check if the total token cost is greater than 0
    cost_caltulator = service_context.callback_manager.handlers[0]
    assert cost_caltulator.total_llm_token_cost > 0
