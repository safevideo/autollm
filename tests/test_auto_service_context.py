from llama_index import Document, ServiceContext, VectorStoreIndex

from autollm.auto.service_context import AutoServiceContext


def test_auto_service_context():
    document = Document.example()

    service_context = AutoServiceContext.from_defaults(enable_cost_calculator=True)

    # Check if the service_context is an instance of ServiceContext
    assert isinstance(service_context, ServiceContext)

    index = VectorStoreIndex.from_documents(documents=[document], service_context=service_context)

    query_engine = index.as_query_engine()

    response = query_engine.query("What is the meaning of life?")

    # Check if the response is not None
    assert response.response is not None

    # Check if the cost calculating handler is working
    cost_caltulator = service_context.callback_manager.handlers[0]
    assert cost_caltulator.total_llm_token_cost > 0
