import os

from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI
from llama_index.query_engine import BaseQueryEngine

from autollm.auto.vector_store_index import AutoVectorStoreIndex

# set the environment variables
azure_api_key = os.environ.get("AZURE_API_KEY")
azure_endpoint = os.environ.get("AZURE_API_BASE")
azure_api_version = os.environ.get("AZURE_API_VERSION")


def test_auto_vector_store():
    documents = [Document.example()]
    llm = AzureOpenAI(
        engine="gpt-35-turbo-1106",
        model="gpt-35-turbo-16k",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="text-embedding-ada-002",
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    vector_store_index = AutoVectorStoreIndex.from_defaults(
        vector_store_type="SimpleVectorStore", documents=documents, service_context=service_context)

    # Check if the vector_store_index is an instance of VectorStoreIndex
    assert isinstance(vector_store_index, VectorStoreIndex)

    query_engine = vector_store_index.as_query_engine()

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)
