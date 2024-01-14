from typing import Optional, Sequence, Union

from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings.utils import EmbedType
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.schema import BaseNode

from autollm.auto.embedding import AutoEmbedding
from autollm.auto.llm import AutoLiteLLM
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store_index import AutoVectorStoreIndex
from autollm.utils.env_utils import load_config_and_dotenv


def create_query_engine(
        documents: Optional[Sequence[Document]] = None,
        nodes: Optional[Sequence[BaseNode]] = None,
        # llm_params
        llm_model: str = "gpt-3.5-turbo",
        llm_max_tokens: Optional[int] = 256,
        llm_temperature: float = 0.1,
        llm_api_base: Optional[str] = None,
        # service_context_params
        system_prompt: str = None,
        query_wrapper_prompt: Union[str, BasePromptTemplate] = None,
        enable_cost_calculator: bool = True,
        embed_model: Optional[str] = "text-embedding-ada-002",
        chunk_size: Optional[int] = 512,
        chunk_overlap: Optional[int] = 100,
        context_window: Optional[int] = None,
        enable_title_extractor: bool = False,
        enable_summary_extractor: bool = False,
        enable_qa_extractor: bool = False,
        enable_keyword_extractor: bool = False,
        enable_entity_extractor: bool = False,
        # query_engine_params
        similarity_top_k: int = 6,
        response_mode: str = "compact",
        refine_prompt: str = None,
        structured_answer_filtering: bool = False,
        # vector_store_params
        vector_store_type: str = "LanceDBVectorStore",
        lancedb_uri: str = "./.lancedb",
        lancedb_table_name: str = "vectors",
        use_async: bool = True,
        exist_ok: bool = False,
        overwrite_existing: bool = False,
        **vector_store_kwargs) -> BaseQueryEngine:
    """
    Create a query engine from parameters.

    Parameters:
        documents (Sequence[Document]): Sequence of llama_index.Document instances.
        nodes (Sequence[BaseNode]): Sequence of llama_index.BaseNode instances.
        llm_model (str): The LLM model to use for the query engine.
        llm_max_tokens (int): The maximum number of tokens to be generated as LLM output.
        llm_temperature (float): The temperature to use for the LLM.
        llm_api_base (str): The API base to use for the LLM.
        system_prompt (str): The system prompt to use for the query engine.
        query_wrapper_prompt (Union[str, BasePromptTemplate]): The query wrapper prompt to use for the query engine.
        enable_cost_calculator (bool): Flag to enable cost calculator logging.
        embed_model (Union[str, EmbedType]): The embedding model to use for generating embeddings. "default" for OpenAI,
                                            "local" for HuggingFace or use full identifier (e.g., local:intfloat/multilingual-e5-large)
        chunk_size (int): The token chunk size for each chunk.
        chunk_overlap (int): The token overlap between each chunk.
        context_window (int): The maximum context size that will get sent to the LLM.
        enable_title_extractor (bool): Flag to enable title extractor.
        enable_summary_extractor (bool): Flag to enable summary extractor.
        enable_qa_extractor (bool): Flag to enable question answering extractor.
        enable_keyword_extractor (bool): Flag to enable keyword extractor.
        enable_entity_extractor (bool): Flag to enable entity extractor.
        similarity_top_k (int): The number of similar documents to return.
        response_mode (str): The response mode to use for the query engine.
        refine_prompt (str): The refine prompt to use for the query engine.
        vector_store_type (str): The vector store type to use for the query engine.
        lancedb_uri (str): The URI to use for the LanceDB vector store.
        lancedb_table_name (str): The table name to use for the LanceDB vector store.

    Returns:
        A llama_index.BaseQueryEngine instance.
    """
    llm = AutoLiteLLM.from_defaults(
        model=llm_model, api_base=llm_api_base, max_tokens=llm_max_tokens, temperature=llm_temperature)

    embedding = AutoEmbedding(model=embed_model)

    service_context = AutoServiceContext.from_defaults(
        llm=llm,
        embed_model=embedding,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        enable_cost_calculator=enable_cost_calculator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        context_window=context_window,
        enable_title_extractor=enable_title_extractor,
        enable_summary_extractor=enable_summary_extractor,
        enable_qa_extractor=enable_qa_extractor,
        enable_keyword_extractor=enable_keyword_extractor,
        enable_entity_extractor=enable_entity_extractor,
    )
    vector_store_index = AutoVectorStoreIndex.from_defaults(
        vector_store_type=vector_store_type,
        lancedb_uri=lancedb_uri,
        lancedb_table_name=lancedb_table_name,
        use_async=use_async,
        documents=documents,
        nodes=nodes,
        service_context=service_context,
        exist_ok=exist_ok,
        overwrite_existing=overwrite_existing,
        **vector_store_kwargs)
    if refine_prompt is not None:
        refine_prompt_template = PromptTemplate(refine_prompt, prompt_type=PromptType.REFINE)
    else:
        refine_prompt_template = None

    # Convert query_wrapper_prompt to PromptTemplate if it is a string
    if isinstance(query_wrapper_prompt, str):
        query_wrapper_prompt = PromptTemplate(template=query_wrapper_prompt)
    response_synthesizer = get_response_synthesizer(
        service_context=service_context,
        text_qa_template=query_wrapper_prompt,
        refine_template=refine_prompt_template,
        response_mode=response_mode,
        structured_answer_filtering=structured_answer_filtering)

    return vector_store_index.as_query_engine(
        similarity_top_k=similarity_top_k, response_synthesizer=response_synthesizer)


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
        embed_model="text-embedding-ada-002",
        chunk_size=512,
        chunk_overlap=None,
        context_window=None,
        # query_engine_params
        similarity_top_k=6,
        # vector_store_params
        vector_store_type="LanceDBVectorStore",
        lancedb_uri="./.lancedb",
        lancedb_table_name="vectors",
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
            nodes: Optional[Sequence[BaseNode]] = None,
            # llm_params
            llm_model: Optional[str] = "gpt-3.5-turbo",
            llm_api_base: Optional[str] = None,
            llm_max_tokens: Optional[int] = None,
            llm_temperature: Optional[float] = 0.1,
            # service_context_params
            system_prompt: str = None,
            query_wrapper_prompt: Union[str, BasePromptTemplate] = None,
            enable_cost_calculator: bool = True,
            embed_model: Optional[str] = "text-embedding-ada-002",
            chunk_size: Optional[int] = 512,
            chunk_overlap: Optional[int] = 200,
            context_window: Optional[int] = None,
            # query_engine_params
            similarity_top_k: int = 6,
            response_mode: str = "compact",
            refine_prompt: str = None,
            structured_answer_filtering: bool = False,
            # vector_store_params
            vector_store_type: str = "LanceDBVectorStore",
            lancedb_uri: str = "./.lancedb",
            lancedb_table_name: str = "vectors",
            use_async: bool = True,
            exist_ok: bool = False,
            overwrite_existing: bool = False,
            **vector_store_kwargs) -> BaseQueryEngine:
        """
        Create an AutoQueryEngine from default parameters.

        Parameters:
            documents (Sequence[Document]): Sequence of llama_index.Document instances.
            nodes (Sequence[BaseNode]): Sequence of llama_index.BaseNode instances.
            llm_model (str): The LLM model to use for the query engine.
            llm_max_tokens (int): The maximum number of tokens to be generated as LLM output.
            llm_temperature (float): The temperature to use for the LLM.
            llm_api_base (str): The API base to use for the LLM.
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (Union[str, BasePromptTemplate]): The query wrapper prompt to use for the query engine.
            enable_cost_calculator (bool): Flag to enable cost calculator logging.
            embed_model (Union[str, EmbedType]): The embedding model to use for generating embeddings.
            chunk_size (int): The token chunk size for each chunk.
            chunk_overlap (int): The token overlap between each chunk.
            context_window (int): The maximum context size that will get sent to the LLM.
            enable_title_extractor (bool): Flag to enable title extractor.
            enable_summary_extractor (bool): Flag to enable summary extractor.
            enable_qa_extractor (bool): Flag to enable question answering extractor.
            enable_keyword_extractor (bool): Flag to enable keyword extractor.
            enable_entity_extractor (bool): Flag to enable entity extractor.
            similarity_top_k (int): The number of similar documents to return.
            response_mode (str): The response mode to use for the query engine.
            refine_prompt (str): The refine prompt to use for the query engine.
            vector_store_type (str): The vector store type to use for the query engine.
            lancedb_uri (str): The URI to use for the LanceDB vector store.
            lancedb_table_name (str): The table name to use for the LanceDB vector store.
            exist_ok (bool): Flag to allow overwriting an existing vector store.
            overwrite_existing (bool): Flag to allow overwriting an existing vector store.

            Returns:
                A llama_index.BaseQueryEngine instance.
        """

        return create_query_engine(
            documents=documents,
            nodes=nodes,
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
            response_mode=response_mode,
            refine_prompt=refine_prompt,
            structured_answer_filtering=structured_answer_filtering,
            # vector_store_params
            vector_store_type=vector_store_type,
            lancedb_uri=lancedb_uri,
            lancedb_table_name=lancedb_table_name,
            use_async=use_async,
            exist_ok=exist_ok,
            overwrite_existing=overwrite_existing,
            **vector_store_kwargs)

    @staticmethod
    def from_config(
            config_file_path: str,
            env_file_path: str = None,
            documents: Sequence[Document] = None,
            nodes: Optional[Sequence[BaseNode]] = None) -> BaseQueryEngine:
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
            nodes=nodes,
            **config,
        )
