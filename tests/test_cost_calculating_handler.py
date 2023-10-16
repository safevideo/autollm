from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.callbacks import CallbackManager

from autollm.callbacks.cost_calculating import CostCalculatingHandler


def test_cost_calculating_handler():
    document = Document.example()

    cost_calculater = CostCalculatingHandler(model="gpt-3.5-turbo", verbose=False)
    callback_manager = CallbackManager([cost_calculater])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    index = VectorStoreIndex.from_documents(documents=[document], service_context=service_context)

    query_engine = index.as_query_engine()

    response = query_engine.query("What is the meaning of life?")
    # Check if the response is not None
    assert response.response is not None

    # Check if the total token cost is greater than 0
    assert cost_calculater.total_llm_token_cost > 0
