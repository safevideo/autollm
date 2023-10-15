import logging
from typing import Sequence

from llama_index import Document
from llama_index.prompts import ChatMessage, ChatPromptTemplate, MessageRole

from autollm.auto.vector_store import AutoVectorStore
from autollm.utils.constants import DEFAULT_INDEX_NAME, DEFAULT_VECTORE_STORE_TYPE
from autollm.utils.hash_utils import check_for_changes
from autollm.utils.templates import QUERY_PROMPT_TEMPLATE, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def set_default_prompt_template() -> ChatPromptTemplate:
    """
    Sets the default prompt template for the query engine.

    Returns:
        SystemPrompt (str): The default system prompt for the query engine.
        ChatPromptTemplate: The default prompt template for the query engine.
    """
    chat_text_msgs = [
        ChatMessage(
            role=MessageRole.USER,
            content=QUERY_PROMPT_TEMPLATE,
        ),
    ]

    return SYSTEM_PROMPT, ChatPromptTemplate(chat_text_msgs)
