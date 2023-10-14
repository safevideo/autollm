import logging

from llama_index import OpenAIEmbedding
from llama_index import ServiceContext as LlamaServiceContext
from llama_index.callbacks import TokenCountingHandler
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms.base import LLM
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole

from autollm.utils.cost_calculation import initialize_token_counting
from autollm.utils.llm_utils import set_default_prompt_template

logger = logging.getLogger(__name__)


class ServiceContext(LlamaServiceContext):
    """
    ServiceContext extends the functionality of LlamaIndex's ServiceContext to include token counting.

    This class is initialized with an optional token_counter and all the parameters required for LlamaIndex's
    ServiceContext. If token_counter is provided, it is used to count the number of tokens used by the LLM.
    """

    def __init__(self, *args, **kwargs):
        self._token_counter: TokenCountingHandler = None
        super().__init__(*args, **kwargs)


class AutoServiceContext:
    """AutoServiceContext extends the functionality of LlamaIndex's ServiceContext to include token
    counting.
    """

    @staticmethod
    def from_defaults(
            llm: LLM,
            embed_model: BaseEmbedding = None,
            enable_cost_logging: bool = False,
            system_prompt: str = None,
            query_wrapper_prompt: str = None,
            *args,
            **kwargs) -> ServiceContext:
        """
        Create a ServiceContext with default parameters with extended enable_token_counting functionality. If
        enable_token_counting is True, tracks the number of tokens used by the LLM for each query.

        Parameters:
            llm (LLM): The LLM to use for the query engine.
            embed_model (BaseEmbedding): The embedding model to use for the query engine.
            enable_cost_logging (bool): Whether to enable cost logging.
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (str): The query wrapper prompt to use for the query engine.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ServiceContext: The initialized ServiceContext from default parameters with extra token counting functionality.
        """
        if not system_prompt or not query_wrapper_prompt:
            logger.info('System prompt and query wrapper prompt not provided. Using default prompts.')
            system_prompt, query_wrapper_prompt = set_default_prompt_template()
        elif isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = ChatPromptTemplate([
                ChatMessage(
                    role=MessageRole.USER,
                    content=query_wrapper_prompt,
                ),
            ])
        else:
            raise ValueError(f'Invalid system_prompt type: {type(query_wrapper_prompt)}')

        if embed_model is None:
            embed_model = OpenAIEmbedding()

        if enable_cost_logging:
            # from your llm_utils module
            token_counter, callback_manager = initialize_token_counting()
            if callback_manager:
                kwargs['callback_manager'] = callback_manager

        service_context: ServiceContext = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, *args, **kwargs)

        if enable_cost_logging:
            service_context._token_counter = token_counter

        return service_context
