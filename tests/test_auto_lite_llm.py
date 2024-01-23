from llama_index.llms import ChatMessage, ChatResponse
from llama_index.llms.base import BaseLLM

from autollm.auto.llm import AutoLiteLLM


def test_auto_lite_llm():
    llm = AutoLiteLLM.from_defaults(model="azure/gpt-35-turbo-1106")

    # Check if the llm is an instance of LLM
    assert isinstance(llm, BaseLLM)

    message = ChatMessage(role="user", content="Hey! how's it going?")
    chat_response = llm.chat([message])

    # Check if the chat response is an instance of ChatResponse
    assert isinstance(chat_response, ChatResponse)
