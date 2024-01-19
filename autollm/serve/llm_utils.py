from typing import Any, Optional

from llama_index.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

from autollm import AutoLiteLLM


class CustomLLM(BaseModel):
    """Data model for custom LLM creation."""

    emoji: str = Field(
        ...,
        description="""
        The emoji to be used when deploying the custom LLM to Hugging Face Spaces.
        """,
        example="📝",
    )
    name: str = Field(
        ...,
        description="""
        The descriptive name of the custom LLM.
        """,
        example="Creative Writing Coach",
    )
    description: str = Field(
        ...,
        description="""
        Very short, one sentence description of what this custom LLM does.
        """,
        example="I'm eager to read your work and give you feedback to improve your skills.",
    )
    instructions: str = Field(
        ...,
        description="""
        Very detailed persona instructions for the custom LLM.
        What does this custom LLM do?
        How does it behave?
        What should it avoid doing?
        How long or short should responses be?
        """,
        example="""
        You are a Creative Writing Coach GPT designed to assist users in enhancing their writing skills.
        You have decades of experience reading creative writing and fiction and giving practical and motivating feedback.
        You offer guidance, suggestions, and constructive criticism to help users refine their prose, poetry,
        or any other form of creative writing. You aim to inspire creativity, help overcome writer's block,
        and provide insights into various writing techniques and styles. You'll start with simple rating of
        your writing and what's good about it before I go into any suggestions. Always be positive and encouraging.
        Ask questions to get more information. Be specific and detailed in your feedback.
        """,
    )


PROMPT_TEMPLATE_STR = """\
Enhance the following user prompt for optimal interaction \
with a custom LLM model. Ensure the revised prompt maintains the \
original intent, is clear and detailed, and is adapted to the \
specific context and task mentioned in the user input.

User Input: {user_prompt}

1. Analyze the basic prompt to understand its primary purpose and context.
2. Refine the prompt to be clear, detailed, specific, and tailored to the context and task.
3. Retain the core elements and intent of the original prompt.
4. Provide an enhanced version of the prompt, ensuring it is optimized for a LLM model interaction.
"""


def create_custom_llm(user_prompt: str, config: Optional[Any] = None) -> CustomLLM:
    """Create a custom LLM using the user prompt."""
    if not user_prompt:
        raise ValueError("Please fill in the area of 'What would you like to make?'")

    llm_model = config.get('llm_model', 'azure/gpt-4-1106')
    llm_max_tokens = config.get('llm_max_tokens', 1024)
    llm_temperature = config.get('llm_temperature', 0.1)
    llm_api_base = config.get('llm_api_base', None)

    llm = AutoLiteLLM.from_defaults(
        model=llm_model,
        max_tokens=llm_max_tokens,
        temperature=llm_temperature,
        api_base=llm_api_base,
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=CustomLLM,
        prompt_template_str=PROMPT_TEMPLATE_STR,
        llm=llm,
        verbose=True,
    )

    output = program(user_prompt=user_prompt)

    return output
