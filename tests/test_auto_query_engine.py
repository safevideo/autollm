import os

from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI
from llama_index.query_engine import BaseQueryEngine

from autollm.auto.query_engine import AutoQueryEngine

# set the environment variables
azure_api_key = os.environ.get("AZURE_API_KEY")
azure_endpoint = os.environ.get("AZURE_API_BASE")
azure_api_version = os.environ.get("AZURE_API_VERSION")

documents = [Document.example()]


def test_auto_query_engine_from_defaults():
    vector_store_type = "SimpleVectorStore"
    query_engine = AutoQueryEngine.from_defaults(
        documents=documents,
        vector_store_type=vector_store_type,
        llm_model="azure/gpt-35-turbo-1106",
        embed_model="azure/text-embedding-ada-002",
    )

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)


def test_auto_query_engine_from_instances():
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
    vector_store_index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context)

    query_engine = AutoQueryEngine.from_instances(
        vector_store_index=vector_store_index, service_context=service_context)
    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)


def test_auto_query_engine_from_config():
    query_engine = AutoQueryEngine.from_config(config_file_path="tests/config.yaml", documents=documents)

    # Check if the query_engine is an instance of BaseQueryEngine
    assert isinstance(query_engine, BaseQueryEngine)
