from llama_index import ServiceContext
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryType
from llama_index.response.schema import RESPONSE_TYPE

from autollm.utils.llm_utils import create_text_qa_template, initialize_token_counting, log_total_cost
from autollm.vectorstores.auto import AutoVectorStore
from autollm.vectorstores.base import BaseVS


class AutoServiceContext(ServiceContext):
    """AutoServiceContext for initializing service context and optionally token counting."""

    def __init__(self, *args, enable_cost_logging=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_counter, self.callback_manager = None, None
        if enable_cost_logging:
            self._token_counter, self.callback_manager = initialize_token_counting()


class QueryEngine:
    """AutoQueryEngine for query execution and optionally logging the query cost."""

    def __init__(self, query_engine: BaseQueryEngine, service_context: AutoServiceContext, **kwargs):
        super().__init__(callback_manager=service_context.callback_manager, **kwargs)
        self._token_counter = service_context._token_counter
        self._query_engine = query_engine

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        if self._token_counter:
            log_total_cost(token_counter=self._token_counter)
            self._token_counter.reset_counts()
        return self._query_engine.query(str_or_query_bundle)


class AutoQueryEngine:
    """AutoQueryEngine for query execution and optionally logging the query cost."""

    def from_defaults(self, vector_store: BaseVS, service_context: AutoServiceContext, qa_teamplet, **kwargs):
        super().__init__(callback_manager=service_context.callback_manager, **kwargs)
        self._token_counter = service_context._token_counter

        self.query_engine = vector_store.vectorindex.as_query_engine(qa_teamplet, service_context)

        return QueryEngine(self.query_engine)


service_context = AutoServiceContext.from_defaults(enable_cost_logging=True)

# Initialize vector store (this could be your AutoVectorStore for example)
vector_store = AutoVectorStore.from_vector_store_type(vector_store_type='qdrant', index_name='quickstart')

# Initialize AutoQueryEngine with QA parameters (replace with actual parameters)
text_qa_template = create_text_qa_template()
query_engine: QueryEngine = AutoQueryEngine(vector_store, service_context, text_qa_template=text_qa_template)

# Make a query
response = query_engine.query('What is the meaning of life?')
