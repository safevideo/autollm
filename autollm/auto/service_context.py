import logging

from llama_index import OpenAIEmbedding
from llama_index import ServiceContext as LlamaServiceContext
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms.base import LLM
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole

from autollm.utils.llm_utils import initialize_token_counting, set_default_prompt_template

logger = logging.getLogger(__name__)


# TODO: update docstring
class ServiceContext(LlamaServiceContext):
    """
    ServiceContext extends the functionality of LlamaIndex's ServiceContext to include token counting.

    This class is initialized with an optional token_counter and all the parameters required for LlamaIndex's
    ServiceContext.
    """

    def __init__(self, *args, **kwargs):
        self._token_counter = None
        super().__init__(*args, **kwargs)


# TODO: update docstring
class AutoServiceContext:

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
        Create an AutoServiceContext from defaults. If enable_cost_logging is True, initializes the token
        counter.

        Parameters:
            enable_cost_logging (bool): Flag to enable cost logging.
            *args, **kwargs: Arguments for the llama_index.ServiceContext class.

        Returns:
            ServiceContext: Initialized service context with token_counter attached.
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


# The idea is to use it like this:
# service_context = AutoServiceContext.from_defaults(enable_cost_logging=True, llm=AutoLLM.from_defaults())
