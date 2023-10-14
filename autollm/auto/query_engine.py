from llama_index import ServiceContext
from llama_index.indices.query.base import BaseQueryEngine

from autollm.auto.llm import AutoLLM
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store import AutoVectorStore
from autollm.vectorstores.base import BaseVS


class AutoQueryEngine:
    """AutoQueryEngine for query execution and optionally logging the query cost."""

    @staticmethod
    def from_instances(vector_store: BaseVS, service_context: ServiceContext, **kwargs) -> BaseQueryEngine:
        """
        Create an AutoQueryEngine from a vector store and a service context.

        Parameters:
            vector_store: Vector store instance.
            service_context: Service context instance.
            **kwargs: Keyword arguments for the query engine.

        Returns:
            An AutoQueryEngine instance.
        """

        return vector_store.vectorindex.as_query_engine(service_context=service_context, **kwargs)

    @staticmethod
    def from_parameters(
            system_prompt: str = None,
            query_wrapper_prompt: str = None,
            enable_cost_calculator: bool = True,
            llm_params: dict = None,
            vector_store_params: dict = {"vector_store_type": "in_memory"},
            service_context_params: dict = None,
            query_engine_params: dict = None) -> BaseQueryEngine:
        """
        Create an AutoQueryEngine from parameters.

        Parameters:
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (str): The query wrapper prompt to use for the query engine.
            enable_cost_calculator (bool): Flag to enable cost calculator logging.
            llm_params (dict): Parameters for the LLM.
            vector_store_params (dict): Parameters for the vector store.
            service_context_params (dict): Parameters for the service context.
            query_engine_params (dict): Parameters for the query engine.

        Returns:
            An AutoQueryEngine instance.
        """

        llm_params = {} if llm_params is None else llm_params
        vector_store_params = {} if vector_store_params is None else vector_store_params
        service_context_params = {} if service_context_params is None else service_context_params
        query_engine_params = {} if query_engine_params is None else query_engine_params

        llm = AutoLLM.from_defaults(**llm_params)
        vector_store = AutoVectorStore.from_defaults(**vector_store_params)
        vector_store.initialize_vectorindex()
        vector_store.connect_vectorstore()
        service_context = AutoServiceContext.from_defaults(
            llm=llm,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            enable_cost_calculator=enable_cost_calculator,
            **service_context_params)

        return vector_store.vectorindex.as_query_engine(
            service_context=service_context, **query_engine_params)
