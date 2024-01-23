import os

from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI

from autollm.auto.service_context import AutoServiceContext

# set the environment variables
azure_api_key = os.environ.get("AZURE_API_KEY")
azure_endpoint = os.environ.get("AZURE_API_BASE")
azure_api_version = os.environ.get("AZURE_API_VERSION")


def test_auto_service_context():
    document = Document.example()
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
    service_context = AutoServiceContext.from_defaults(
        enable_cost_calculator=True, llm=llm, embed_model=embed_model)

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
