from typing import Optional, Union

from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager
from llama_index.embeddings.utils import EmbedType
from llama_index.extractors import (
    EntityExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.llms.utils import LLMType
from llama_index.prompts import PromptTemplate
from llama_index.prompts.base import BasePromptTemplate
from llama_index.text_splitter import SentenceSplitter

from autollm.callbacks.cost_calculating import CostCalculatingHandler
from autollm.utils.llm_utils import set_default_prompt_template


class AutoServiceContext:
    """AutoServiceContext extends the functionality of LlamaIndex's ServiceContext to include token
    counting.
    """

    @staticmethod
    def from_defaults(
            llm: Optional[LLMType] = "default",
            embed_model: Optional[EmbedType] = "default",
            system_prompt: str = None,
            query_wrapper_prompt: Union[str, BasePromptTemplate] = None,
            enable_cost_calculator: bool = False,
            chunk_size: Optional[int] = 512,
            chunk_overlap: Optional[int] = 200,
            context_window: Optional[int] = None,
            enable_title_extractor: bool = False,
            enable_summary_extractor: bool = False,
            enable_qa_extractor: bool = False,
            enable_keyword_extractor: bool = False,
            enable_entity_extractor: bool = False,
            **kwargs) -> ServiceContext:
        """
        Create a ServiceContext with default parameters with extended enable_token_counting functionality. If
        enable_token_counting is True, tracks the number of tokens used by the LLM for each query.

        Parameters:
            llm (LLM): The LLM to use for the query engine. Defaults to gpt-3.5-turbo.
            embed_model (BaseEmbedding): The embedding model to use for the query engine. Defaults to OpenAIEmbedding.
            system_prompt (str): The system prompt to use for the query engine.
            query_wrapper_prompt (Union[str, BasePromptTemplate]): The query wrapper prompt to use for the query engine.
            enable_cost_calculator (bool): Flag to enable cost calculator logging.
            chunk_size (int): The token chunk size for each chunk.
            chunk_overlap (int): The token overlap between each chunk.
            context_window (int): The maximum context size that will get sent to the LLM.
            enable_title_extractor (bool): Flag to enable title extractor.
            enable_summary_extractor (bool): Flag to enable summary extractor.
            enable_qa_extractor (bool): Flag to enable question answering extractor.
            enable_keyword_extractor (bool): Flag to enable keyword extractor.
            enable_entity_extractor (bool): Flag to enable entity extractor.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ServiceContext: The initialized ServiceContext from default parameters with extra token counting functionality.
        """
        if not system_prompt and not query_wrapper_prompt:
            system_prompt, query_wrapper_prompt = set_default_prompt_template()
        # Convert system_prompt to ChatPromptTemplate if it is a string
        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(template=query_wrapper_prompt)

        callback_manager: CallbackManager = kwargs.get('callback_manager', CallbackManager())
        if enable_cost_calculator:
            llm_model_name = llm.metadata.model_name if not "default" else "gpt-3.5-turbo"
            callback_manager.add_handler(CostCalculatingHandler(model_name=llm_model_name, verbose=True))

        sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        transformations = [sentence_splitter]
        if enable_entity_extractor:
            transformations.append(EntityExtractor())
        if enable_keyword_extractor:
            transformations.append(KeywordExtractor(llm=llm, keywords=5))
        if enable_summary_extractor:
            transformations.append(SummaryExtractor(llm=llm, summaries=["prev", "self"]))
        if enable_title_extractor:
            transformations.append(TitleExtractor(llm=llm, nodes=5))
        if enable_qa_extractor:
            transformations.append(QuestionsAnsweredExtractor(llm=llm, questions=5))

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            transformations=transformations,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_window=context_window,
            callback_manager=callback_manager,
            **kwargs)

        return service_context
