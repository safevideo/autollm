from typing import Optional, Sequence, Union

from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings.utils import EmbedType
from llama_index.indices.query.base import BaseQueryEngine

from autollm.auto.llm import AutoLiteLLM
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store_index import AutoVectorStoreIndex
from autollm.utils.env_utils import load_config_and_dotenv


def create_query_engine(
        documents: Optional[Sequence[Document]] = None,
        # llm_params
        llm_model: str = "gpt-3.5-turbo",
        llm_max_tokens: Optional[int] = 256,
        llm_temperature: float = 0.1,
        llm_api_base: Optional[str] = None,
        # service_context_params
        system_prompt: str = None,
        query_wrapper_prompt: str = None,
        enable_cost_calculator: bool = True,
        embed_model: Union[str, EmbedType] = "default",  # ["default", "local"]
        chunk_size: Optional[int] = 512,
        chunk_overlap: Optional[int] = None,
        context_window: Optional[int] = None,
        # query_engine_params
        similarity_top_k: int = 6,
        # vector_store_params
        vector_store_type: str = "LanceDBVectorStore",
        lancedb_uri: str = "./.lancedb",
        lancedb_table_name: str = "vectors",
        enable_metadata_extraction: bool = False,
        **vector_store_kwargs) -> BaseQueryEngine:
    """
    Create a query engine from parameters.

    Parameters:
        documents (Sequence[Document]): Sequence of llama_index.Document instances.
        system_prompt (str): The system prompt to use for the query engine.
        query_wrapper_prompt (str): The query wrapper prompt to use for the query engine.
        enable_cost_calculator (bool): Flag to enable cost calculator logging.
        embed_model (Union[str, EmbedType]): The embedding model to use for generating embeddings. "default" for OpenAI,
                                            "local" for HuggingFace or use full identifier (e.g., local:intfloat/multilingual-e5-large)
        llm_params (dict): Parameters for the LLM.
        vector_store_params (dict): Parameters for the vector store.
        service_context_params (dict): Parameters for the service context.
        query_engine_params (dict): Parameters for the query engine.

    Returns:
        A llama_index.BaseQueryEngine instance.
    """
    llm = AutoLiteLLM.from_defaults(
        model=llm_model, api_base=llm_api_base, max_tokens=llm_max_tokens, temperature=llm_temperature)
    service_context = AutoServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        enable_cost_calculator=enable_cost_calculator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        context_window=context_window)
    vector_store_index = AutoVectorStoreIndex.from_defaults(
        vector_store_type=vector_store_type,
        lancedb_uri=lancedb_uri,
        lancedb_table_name=lancedb_table_name,
        enable_metadata_extraction=enable_metadata_extraction,
        documents=documents,
        service_context=service_context,
        **vector_store_kwargs)

    return vector_store_index.as_query_engine(similarity_top_k=similarity_top_k)


class AutoQueryEngine:
    """
    AutoQueryEngine for query execution and optionally logging the query cost.

    ```python
    from autollm.auto.query_engine import AutoQueryEngine

    # Create an AutoQueryEngine from a config file and .env file
    query_engine = AutoQueryEngine.from_config("config.yaml", ".env")

    # Create an AutoQueryEngine from a vector store index and service context
    query_engine = AutoQueryEngine.from_instances(vector_store_index, service_context)

    # Create an AutoQueryEngine from defaults
    query_engine = AutoQueryEngine.from_defaults(
        documents=documents,
        # llm_params
        llm_model="gpt-3.5-turbo",
        llm_api_base=None,
        llm_max_tokens=None,
        llm_temperature=0.1,
        # service_context_params
        system_prompt=None,
        query_wrapper_prompt=None,
        enable_cost_calculator=True,
        embed_model="default",  # ["default", "local"]
        chunk_size=512,
        chunk_overlap=None,
        context_window=None,
        # query_engine_params
        similarity_top_k=6,
        # vector_store_params
        vector_store_type="LanceDBVectorStore",
        lancedb_uri="./.lancedb",
        lancedb_table_name="vectors",
        enable_metadata_extraction=False,
        **vector_store_kwargs)
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
    def from_defaults(
            documents: Optional[Sequence[Document]] = None,
            # llm_params
            llm_model: str = "gpt-3.5-turbo",
            llm_api_base: Optional[str] = None,
            llm_max_tokens: Optional[int] = None,
            llm_temperature: float = 0.1,
            # service_context_params
            system_prompt: str = None,
            query_wrapper_prompt: str = None,
            enable_cost_calculator: bool = True,
            embed_model: Union[str, EmbedType] = "default",  # ["default", "local"]
            chunk_size: Optional[int] = 512,
            chunk_overlap: Optional[int] = None,
            context_window: Optional[int] = None,
            # query_engine_params
            similarity_top_k: int = 6,
            # vector_store_params
            vector_store_type: str = "LanceDBVectorStore",
            lancedb_uri: str = "./.lancedb",
            lancedb_table_name: str = "vectors",
            enable_metadata_extraction: bool = False,
            **vector_store_kwargs) -> BaseQueryEngine:
        """
        Create an AutoQueryEngine from default parameters.

        Parameters:
            documents (Sequence[Document]): Sequence of llama_index.Document instances.
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (str): The query wrapper prompt to use for the query engine.
            enable_cost_calculator (bool): Flag to enable cost calculator logging.
            embed_model (Union[str, EmbedType]): The embedding model to use for generating embeddings. "default" for OpenAI,
                                                "local" for HuggingFace or use full identifier (e.g., local:intfloat/multilingual-e5-large)
            llm_params (dict): Parameters for the LLM.
            vector_store_params (dict): Parameters for the vector store.
            service_context_params (dict): Parameters for the service context.
            query_engine_params (dict): Parameters for the query engine.

        Returns:
            A llama_index.BaseQueryEngine instance.
        """

        return create_query_engine(
            documents=documents,
            # llm_params
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            # service_context_params
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            enable_cost_calculator=enable_cost_calculator,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_window=context_window,
            # query_engine_params
            similarity_top_k=similarity_top_k,
            # vector_store_params
            vector_store_type=vector_store_type,
            lancedb_uri=lancedb_uri,
            lancedb_table_name=lancedb_table_name,
            enable_metadata_extraction=enable_metadata_extraction,
            **vector_store_kwargs)

    @staticmethod
    def from_parameters(
            documents: Sequence[Document] = None,
            system_prompt: str = None,
            query_wrapper_prompt: str = None,
            enable_cost_calculator: bool = True,
            embed_model: Union[str, EmbedType] = "default",  # ["default", "local"]
            llm_params: dict = None,
            vector_store_params: dict = None,
            service_context_params: dict = None,
            query_engine_params: dict = None) -> BaseQueryEngine:
        """
        DEPRECATED. Use AutoQueryEngine.from_defaults instead.

        Create an AutoQueryEngine from parameters.

        Parameters:
            documents (Sequence[Document]): Sequence of llama_index.Document instances.
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (str): The query wrapper prompt to use for the query engine.
            enable_cost_calculator (bool): Flag to enable cost calculator logging.
            embed_model (Union[str, EmbedType]): The embedding model to use for generating embeddings. "default" for OpenAI,
                                                "local" for HuggingFace or use full identifier (e.g., local:intfloat/multilingual-e5-large)
            llm_params (dict): Parameters for the LLM.
            vector_store_params (dict): Parameters for the vector store.
            service_context_params (dict): Parameters for the service context.
            query_engine_params (dict): Parameters for the query engine.

        Returns:
            A llama_index.BaseQueryEngine instance.
        """

        # TODO: Remove this method in the next release
        raise ValueError(
            "AutoQueryEngine.from_parameters is deprecated. Use AutoQueryEngine.from_defaults instead.")

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
            llm_model=config.get('llm_model'),
            llm_api_base=config.get('llm_api_base'),
            llm_max_tokens=config.get('llm_max_tokens'),
            llm_temperature=config.get('llm_temperature'),
            system_prompt=config.get('system_prompt'),
            query_wrapper_prompt=config.get('query_wrapper_prompt'),
            enable_cost_calculator=config.get('enable_cost_calculator'),
            embed_model=config.get('embed_model'),
            chunk_size=config.get('chunk_size'),
            chunk_overlap=config.get('chunk_overlap'),
            context_window=config.get('context_window'),
            similarity_top_k=config.get('similarity_top_k'),
            vector_store_type=config.get('vector_store_type'),
            lancedb_uri=config.get('lancedb_uri'),
            lancedb_table_name=config.get('lancedb_table_name'),
            enable_metadata_extraction=config.get('enable_metadata_extraction'),
            **config.get('vector_store_kwargs', {}))
