from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.llms.base import LLM
from llama_index.query_engine import BaseQueryEngine

from autollm.auto.llm import AutoLiteLLM


def test_auto_lite_llm():
    document = Document.example()

    llm = AutoLiteLLM.from_defaults(model="gpt-3.5-turbo")

    # Check if the llm is an instance of LLM
    assert isinstance(llm, LLM)

    service_context = ServiceContext.from_defaults(llm=llm)

    index = VectorStoreIndex.from_documents(documents=[document], service_context=service_context)

    query_engine = index.as_query_engine()

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)
