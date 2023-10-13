from llama_index import ServiceContext
from llama_index.indices import BaseQueryEngine, QueryType, RESPONSE_TYPE, QueryBundle
from .llm_utils import initialize_token_counting, log_total_cost, create_text_qa_template
from vectorstores.auto_vector_store import AutoVectorStore

# Assuming initialize_token_counting and log_total_cost are imported from your llm_utils

class AutoServiceContext(ServiceContext):
    """AutoServiceContext for initializing service context and optionally token counting."""
    def __init__(self, *args, enable_cost_logging=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_counter, self.callback_manager = None, None
        if enable_cost_logging:
            self._token_counter, self.callback_manager = initialize_token_counting()

class AutoQueryEngine(BaseQueryEngine):
    """AutoQueryEngine for query execution and optionally logging the query cost."""
    def __init__(self, vector_store, service_context: AutoServiceContext, **kwargs):
        super().__init__(callback_manager=service_context.callback_manager, **kwargs)
        self._token_counter = service_context._token_counter
        self.vector_store = vector_store

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        if self._token_counter:
            log_total_cost(token_counter=self._token_counter)
            self._token_counter.reset_counts()
        with self.callback_manager.as_trace("query"):
            if isinstance(str_or_query_bundle, str):
                str_or_query_bundle = QueryBundle(str_or_query_bundle)
            response = self._query(str_or_query_bundle)
            return response

# Usage Example
# Initialize AutoServiceContext
service_context = AutoServiceContext.from_defaults(enable_cost_logging=True)

# Initialize vector store (this could be your AutoVectorStore for example)
vector_store = AutoVectorStore.from_vector_store_type(vector_store_type="qdrant", index_name="quickstart")

# Initialize AutoQueryEngine with QA parameters (replace with actual parameters)
text_qa_template = create_text_qa_template()
query_engine = AutoQueryEngine(vector_store, service_context, text_qa_template=text_qa_template)

# Make a query
response = query_engine.query("What is the meaning of life?")
