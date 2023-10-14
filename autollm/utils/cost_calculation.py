# Description: Utility functions for calculating the cost of a query.
#
# This file contains functions for calculating the cost of a query based on the number of tokens used by the LLM.
# The cost is calculated based on the number of tokens used by the LLM for the prompt and completion.
# The cost is calculated based on the model used for encoding the prompt and completion.

import logging

import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler

logger = logging.getLogger(__name__)

MODEL_COST = {
    'OpenAI': {
        'gpt-3.5-turbo': {
            'prompt': {
                'price': 0.0015,
                'unit': 1000
            },
            'completion': {
                'price': 0.002,
                'unit': 1000
            }
        },
        'gpt-4': {
            'prompt': {
                'price': 0.03,
                'unit': 1000
            },
            'completion': {
                'price': 0.06,
                'unit': 1000
            }
        }
    }
    # TODO: add more models (aws bedrock: https://aws.amazon.com/tr/bedrock/pricing/, anyscale: https://docs.endpoints.anyscale.com/pricing/#pricing, palm: https://cloud.google.com/vertex-ai/docs/generative-ai/pricing, azure-openai: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
}


# TODO: add llm_class_name input, update docstring
def initialize_token_counting(encoding_model: str = 'gpt-3.5-turbo'):
    """
    Initializes the Token Counting Handler for tracking token usage.

    Parameters:
        encoding_model (str): The name of the encoding model to use for tokenization.

    Returns:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler for counting tokens in LLM and embedding events.
        callback_manager (CallbackManager): Callback Manager initialized with the Token Counting Handler.
    """
    # Initialize the Token Counting Handler
    token_counter = generate_token_counter(encoding_model=encoding_model)

    # Initialize Callback Manager and add Token Counting Handler
    callback_manager = CallbackManager([token_counter])

    return token_counter, callback_manager


def generate_token_counter(encoding_model: str = 'gpt-3.5-turbo'):
    """
    Generates a Token Counting Handler for tracking token usage.

    Parameters:
        encoding_model (str): The name of the encoding model to use.

    Returns:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
    """
    # Initialize the Token Counting Handler
    tokenizer = tiktoken.encoding_for_model(encoding_model).encode  # Update the model name as needed
    token_counter = TokenCountingHandler(tokenizer=tokenizer, verbose=True)

    return token_counter


def calculate_total_cost(
        token_counter: TokenCountingHandler, llm_class_name: str, model_name: str = 'gpt-3.5-turbo'):
    """
    Calculate the total cost based on the model and token usage for both the prompt and completion.

    Parameters:
        token_counter (TokenCountingHandler): Token Counting Handler initialized with the tokenizer.
        llm_class_name (str): The class name of the Language Learning Model being used.
        model_name (str): The name of the model being used.

    Returns:
        total_cost (float): The total cost of the query in USD, based on the token usage and model.
    """
    # Access cost information for the given LLM class and model
    llm_cost_info = MODEL_COST.get(llm_class_name)
    model_cost_info = llm_cost_info.get(model_name) if llm_cost_info else None

    # Raise error if cost information is not available
    if not llm_cost_info or not model_cost_info:
        available_llm_classes = ", ".join(MODEL_COST.keys())
        available_models = ", ".join(llm_cost_info.keys()) if llm_cost_info else "None"
        logger.warning(
            f"Cost information for model {model_name} under LLM class {llm_class_name} is not available.\n"
            f"Available LLM classes: {available_llm_classes}\n"
            f"Available models for LLM class {llm_class_name}: {available_models}\n"
            "Disabling cost calculation.")
        total_cost = '-'
    else:
        prompt_token_count = token_counter.prompt_llm_token_count
        completion_token_count = token_counter.completion_llm_token_count

        prompt_cost = (prompt_token_count /
                       model_cost_info['prompt']['unit']) * model_cost_info['prompt']['price']
        completion_cost = (completion_token_count /
                           model_cost_info['completion']['unit']) * model_cost_info['completion']['price']

        total_cost = prompt_cost + completion_cost

    return total_cost


def log_total_cost(token_counter: TokenCountingHandler):
    """
    Logs the total cost based on token usage for both the prompt and completion.

    Parameters:
        token_counter (TokenCountingHandler): Initialized Token Counting Handler.
    """
    total_cost = calculate_total_cost(token_counter)
    logger.info(f'Total cost for this query: {total_cost} USD')
    token_counter.reset_counts()
