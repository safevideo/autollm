from typing import Optional

from llama_index.llms import LiteLLM
from llama_index.llms.base import BaseLLM


class AutoLiteLLM:
    """AutoLiteLLM lets you dynamically initialize any LLM based on the llm class name and additional
    parameters.
    """

    @staticmethod
    def from_defaults(
            model: str = "gpt-3.5-turbo",
            max_tokens: Optional[int] = 256,
            temperature: float = 0.1,
            system_prompt: Optional[str] = None,
            api_base: Optional[str] = None) -> BaseLLM:
        """
        Create any LLM by model name. Check https://docs.litellm.ai/docs/providers for a list of
        supported models.

        If an argument is specified, then use the argument value provided for that
        parameter. If an argument is not specified, then use the default value.

        Parameters:
            model: Name of the LLM model to be initialized. Check
        https://docs.litellm.ai/docs/providers for a list of supported models.
            max_tokens: The maximum number of tokens to generate by the LLM.
            temperature: The temperature to use when sampling from the distribution.
            system_prompt: The system prompt to use for the LLM.
            api_base: The API base URL to use for the LLM.

        Returns:
            LLM: The initialized LiteLLM instance for given model name and parameter set.
        """

        return LiteLLM(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            api_base=api_base)
