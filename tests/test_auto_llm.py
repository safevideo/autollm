from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.llms.base import LLM

from autollm.auto.llm import AutoLLM


def test_auto_llm():
    document = Document.example()

    llm = AutoLLM.from_defaults(model="gpt-3.5-turbo")

    # Check if the llm is an instance of LLM
    assert isinstance(llm, LLM)

    service_context = ServiceContext.from_defaults(llm=llm)

    index = VectorStoreIndex.from_documents(documents=[document], service_context=service_context)

    query_engine = index.as_query_engine()

    response = query_engine.query("What is the meaning of life?")

    # Check if the response is not None
    assert response.response is not None
