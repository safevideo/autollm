import logging
from typing import Union

from llama_index import OpenAIEmbedding, ServiceContext
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms.utils import LLMType
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole
from llama_index.prompts.base import BasePromptTemplate

from autollm.callbacks.cost_calculating import CostCalculatingHandler
from autollm.utils.llm_utils import set_default_prompt_template

logger = logging.getLogger(__name__)


class AutoServiceContext:
    """AutoServiceContext extends the functionality of LlamaIndex's ServiceContext to include token
    counting.
    """

    @staticmethod
    def from_defaults(
            llm: LLMType = "default",
            embed_model: BaseEmbedding = None,
            system_prompt: str = None,
            query_wrapper_prompt: Union[str, BasePromptTemplate] = None,
            enable_cost_calculator: bool = False,
            **kwargs) -> ServiceContext:
        """
        Create a ServiceContext with default parameters with extended enable_token_counting functionality. If
        enable_token_counting is True, tracks the number of tokens used by the LLM for each query.

        Parameters:
            llm (LLM): The LLM to use for the query engine. Defaults to gp3-5-turbo.
            embed_model (BaseEmbedding): The embedding model to use for the query engine.
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (Union[str, BasePromptTemplate]): The query wrapper prompt to use for the query engine.
            cost_calculator_verbose (bool): Flag to enable cost calculator logging.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ServiceContext: The initialized ServiceContext from default parameters with extra token counting functionality.
        """
        if not system_prompt or not query_wrapper_prompt:
            logger.info('System prompt and query wrapper prompt not provided. Using default prompts.')
            system_prompt, query_wrapper_prompt = set_default_prompt_template()
        # Convert system_prompt to ChatPromptTemplate if it is a string
        elif isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = ChatPromptTemplate([
                ChatMessage(
                    role=MessageRole.USER,
                    content=query_wrapper_prompt,
                ),
            ])
        # Use the provided query wrapper prompt as is if it is a BasePromptTemplate
        elif isinstance(query_wrapper_prompt, BasePromptTemplate):
            pass
        else:
            raise ValueError(f'Invalid system_prompt type: {type(query_wrapper_prompt)}')

        callback_manager: CallbackManager = kwargs.get('callback_manager', CallbackManager())
        if enable_cost_calculator:
            model = llm.metadata.model_name if not "default" else "gpt-3.5-turbo"
            callback_manager.add_handler(CostCalculatingHandler(model=model, verbose=True))

        if embed_model is None:
            embed_model = OpenAIEmbedding()

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            callback_manager=callback_manager,
            **kwargs)

        return service_context
