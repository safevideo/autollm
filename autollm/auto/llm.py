from typing import Any, Dict

from llama_index.llms import OpenAI
from llama_index.llms.base import LLM

from autollm.auto.service_context import AutoServiceContext
from autollm.utils.constants import DEFAULT_LLM_CLASS_NAME, DEFAULT_OPENAI_MODEL


# TODO: update docstring
def import_llm_class(llm_class_name: str):
    """
    Imports a predefined vector store class by class name.

    Args:

    Returns:
        The imported VectorStore class.
    """
    module = __import__("llama_index.llms", fromlist=[llm_class_name])
    class_ = getattr(module, llm_class_name)
    return class_


def import_langchainlm_with_bedrock(
        client: Any,
        model: str,
        region_name: str | None = None,
        credentials_profile_name: str | None = None,
        model_kwargs: Dict | None = None,
        endpoint_url: str | None = None,
        streaming: bool = False,
        **kwargs) -> LLM:
    """
    Bedrock models.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Bedrock service.
    """

    from langchain.llms.bedrock import Bedrock
    from llama_index.llms.langchain import LangChainLLM

    return LangChainLLM(
        Bedrock(
            client=client,
            region_name=region_name,
            credentials_profile_name=credentials_profile_name,
            model_id=model,
            model_kwargs=model_kwargs,
            endpoint_url=endpoint_url,
            streaming=streaming,
            **kwargs))


# TODO: update docstring
class AutoLLM:

    @staticmethod
    def from_defaults(llm_class_name: str = DEFAULT_LLM_CLASS_NAME, *args, **kwargs) -> LLM:
        """
        Create an LLM from defaults. If enable_cost_logging is True, initializes the token counter.

        Parameters:
            llm_class_name (str): Name of the llama_index LLM class to be initialized.
            enable_cost_logging (bool): Flag to enable cost logging.
            *args, **kwargs: Arguments for the llama_index.LLM class.

        Returns:
            LLM: Initialized LLM with token_counter attached.
        """
        if llm_class_name == 'Bedrock':
            llm = import_langchainlm_with_bedrock(*args, **kwargs)
        else:
            llm_class = import_llm_class(llm_class_name=llm_class_name)
            if isinstance(llm_class, OpenAI):
                if 'model' not in kwargs:
                    kwargs['model'] = DEFAULT_OPENAI_MODEL
            llm = llm_class(*args, **kwargs)

        return llm
