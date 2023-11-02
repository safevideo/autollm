from typing import Optional, Sequence

from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings.utils import EmbedType
from llama_index.indices.query.base import BaseQueryEngine

from autollm.auto.llm import AutoLLM
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store_index import AutoVectorStoreIndex
from autollm.utils.env_utils import load_config_and_dotenv


def create_query_engine(
        documents: Sequence[Document] = None,
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        enable_cost_calculator: bool = True,
        embed_model: Optional[EmbedType] = "default",
        llm_params: dict = None,
        vector_store_params: dict = None,
        service_context_params: dict = None,
        query_engine_params: dict = None) -> BaseQueryEngine:
    """
    Create a query engine from parameters.

    Parameters:
        documents (Sequence[Document]): Sequence of llama_index.Document instances.
        system_prompt (str): The system prompt to use for the query engine.
        query_wrapper_prompt (str): The query wrapper prompt to use for the query engine.
        enable_cost_calculator (bool): Flag to enable cost calculator logging.
        embed_model (BaseEmbedding): The embedding model to use for the query engine. Defaults to OpenAIEmbedding.
        llm_params (dict): Parameters for the LLM.
        vector_store_params (dict): Parameters for the vector store.
        service_context_params (dict): Parameters for the service context.
        query_engine_params (dict): Parameters for the query engine.

    Returns:
        A llama_index.BaseQueryEngine instance.
    """

    llm_params = {} if llm_params is None else llm_params
    vector_store_params = {
        "vector_store_type": "LanceDBVectorStore"
    } if vector_store_params is None else vector_store_params
    service_context_params = {} if service_context_params is None else service_context_params
    query_engine_params = {} if query_engine_params is None else query_engine_params

    llm = AutoLLM.from_defaults(**llm_params)
    service_context = AutoServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        enable_cost_calculator=enable_cost_calculator,
        **service_context_params)
    vector_store_index = AutoVectorStoreIndex.from_defaults(
        **vector_store_params, documents=documents, service_context=service_context)

    return vector_store_index.as_query_engine(**query_engine_params)


class AutoQueryEngine:
    """
    AutoQueryEngine for query execution and optionally logging the query cost.

    ```python
    from autollm.auto.query_engine import AutoQueryEngine

    # Create an AutoQueryEngine from a config file and .env file
    query_engine = AutoQueryEngine.from_config("config.yaml", ".env")

    # Create an AutoQueryEngine from a vector store index and service context
    query_engine = AutoQueryEngine.from_instances(vector_store_index, service_context)

    # Create an AutoQueryEngine from parameters
    query_engine = AutoQueryEngine.from_parameters(
      documents=documents,
      system_prompt=system_prompt,
      query_wrapper_prompt=query_wrapper_prompt,
      enable_cost_calculator=enable_cost_calculator,
      llm_params=llm_params,
      vector_store_params=vector_store_params,
      service_context_params=service_context_params,
      query_engine_params=query_engine_params
    )
    ```
    """

    @staticmethod
    def from_instances(
            vector_store_index: VectorStoreIndex, service_context: ServiceContext,
            **kwargs) -> BaseQueryEngine:
        """
        Create an AutoQueryEngine from a vector store index and a service context.

        Parameters:
            vector_store_index: llama_index.VectorStoreIndex instance.
            service_context: llama_index.ServiceContext instance.
            **kwargs: Keyword arguments for the query engine.

        Returns:
            A llama_index.BaseQueryEngine instance.
        """

        return vector_store_index.as_query_engine(service_context=service_context, **kwargs)

    @staticmethod
    def from_parameters(
            documents: Sequence[Document] = None,
            system_prompt: str = None,
            query_wrapper_prompt: str = None,
            enable_cost_calculator: bool = True,
            llm_params: dict = None,
            vector_store_params: dict = None,
            service_context_params: dict = None,
            query_engine_params: dict = None) -> BaseQueryEngine:
        """
        Create an AutoQueryEngine from parameters.

        Parameters:
            documents (Sequence[Document]): Sequence of llama_index.Document instances.
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (str): The query wrapper prompt to use for the query engine.
            enable_cost_calculator (bool): Flag to enable cost calculator logging.
            llm_params (dict): Parameters for the LLM.
            vector_store_params (dict): Parameters for the vector store.
            service_context_params (dict): Parameters for the service context.
            query_engine_params (dict): Parameters for the query engine.

        Returns:
            A llama_index.BaseQueryEngine instance.
        """

        return create_query_engine(
            documents=documents,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            enable_cost_calculator=enable_cost_calculator,
            llm_params=llm_params,
            vector_store_params=vector_store_params,
            service_context_params=service_context_params,
            query_engine_params=query_engine_params)

    @staticmethod
    def from_config(
            config_file_path: str,
            env_file_path: str = None,
            documents: Sequence[Document] = None) -> BaseQueryEngine:
        """
        Create an AutoQueryEngine from a config file and optionally a .env file.

        Parameters:
            config_file_path (str): Path to the YAML configuration file.
            env_file_path (str): Path to the .env file.
            documents (Sequence[Document]): Sequence of llama_index.Document instances.

        Returns:
            A llama_index.BaseQueryEngine instance.
        """

        config = load_config_and_dotenv(config_file_path, env_file_path)
        # Get the first task configuration
        config = config['tasks'][0]

        return create_query_engine(
            documents=documents,
            system_prompt=config.get('system_prompt'),
            query_wrapper_prompt=config.get('query_wrapper_prompt'),
            enable_cost_calculator=config.get('enable_cost_calculator'),
            llm_params=config.get('llm_params'),
            vector_store_params=config.get('vector_store_params'),
            service_context_params=config.get('service_context_params'),
            query_engine_params=config.get('query_engine_params'))
