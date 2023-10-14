from llama_index import ServiceContext
from llama_index.indices.query.base import BaseQueryEngine

from autollm.auto.llm import AutoLLM
from autollm.auto.service_context import AutoServiceContext
from autollm.auto.vector_store import AutoVectorStore
from autollm.vectorstores.base import BaseVS


class AutoQueryEngine:
    """AutoQueryEngine for query execution and optionally logging the query cost."""

    @staticmethod  # TODO: update docstring
    def from_instances(vector_store: BaseVS, service_context: ServiceContext, **kwargs) -> BaseQueryEngine:

        return vector_store.vectorindex.as_query_engine(service_context=service_context, **kwargs)

    @staticmethod  # TODO: update docstring
    def from_parameters(
            system_prompt: str = None,
            query_wrapper_prompt: str = None,
            enable_cost_calculator: bool = True,
            llm_params: dict = None,
            vector_store_params: dict = {"vector_store_type": "in_memory"},
            service_context_params: dict = None,
            query_engine_params: dict = None) -> BaseQueryEngine:

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
