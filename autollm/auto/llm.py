from llama_index.llms import LiteLLM
from llama_index.llms.base import LLM


class AutoLLM:
    """AutoLLM lets you dynamically initialize any LLM based on the llm class name and additional
    parameters.
    """

    @staticmethod
    def from_defaults(model: str = "gpt-3.5-turbo", *args, **kwargs) -> LLM:
        """
        Create any LLM by model name. Check https://docs.litellm.ai/docs/providers for a list of
        supported models.

        If an argument is specified, then use the argument value provided for that
        parameter. If an argument is not specified, then use the default value.

        Parameters:
            model: Name of the LLM model to be initialized. Check
        https://docs.litellm.ai/docs/providers for a list of supported models.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            LLM: The initialized LiteLLM instance for given model name and parameter set.
        """

        return LiteLLM(model=model, *args, **kwargs)
