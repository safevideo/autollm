from llama_index.callbacks import TokenCountingHandler
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryType
from llama_index.response.schema import RESPONSE_TYPE

from autollm.auto.service_context import ServiceContext
from autollm.utils.cost_calculation import log_total_cost
from autollm.vectorstores.base import BaseVS


class QueryEngine(BaseQueryEngine):
    """AutoQueryEngine for query execution and optionally logging the query cost."""

    def __init__(self, query_engine: BaseQueryEngine, service_context: ServiceContext, **kwargs):
        self._service_context = service_context
        self._query_engine = query_engine

    @property
    def _token_counter(self) -> TokenCountingHandler:
        return self._service_context._token_counter

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        response = self._query_engine.query(str_or_query_bundle)
        if self._token_counter:
            log_total_cost(token_counter=self._token_counter)
        return response


class AutoQueryEngine:
    """AutoQueryEngine for query execution and optionally logging the query cost."""

    def from_defaults(self, vector_store: BaseVS, service_context: ServiceContext, **kwargs):
        super().__init__(callback_manager=service_context.callback_manager, **kwargs)
        self._token_counter = service_context._token_counter

        self.query_engine = vector_store.vectorindex.as_query_engine(service_context=service_context)

        return QueryEngine(self.query_engine)
