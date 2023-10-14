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
            cost_calculator_verbose: bool = True,
            llm_params: dict = None,
            vector_store_params: dict = None,
            service_context_params: dict = None,
            query_engine_params: dict = None) -> BaseQueryEngine:

        llm = AutoLLM.from_defaults(**llm_params)
        vector_store = AutoVectorStore.from_defaults(**vector_store_params)
        service_context = AutoServiceContext.from_defaults(
            llm=llm,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            cost_calculator_verbose=cost_calculator_verbose,
            **service_context_params)

        return vector_store.vectorindex.as_query_engine(
            service_context=service_context, **query_engine_params)
