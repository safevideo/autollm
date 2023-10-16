import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from litellm.utils import cost_per_token, token_counter
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.callbacks.token_counting import TokenCountingEvent, TokenCountingHandler

logger = logging.getLogger(__name__)


@dataclass
class CostCalculatingEvent:
    prompt: str
    prompt_token_cost: int
    completion: str
    completion_token_cost: int
    total_token_cost: int = 0
    event_id: str = ""

    def __post_init__(self) -> None:
        self.total_token_cost = self.prompt_token_cost + self.completion_token_cost


def get_llm_token_counts(
        payload: Dict[str, Any], event_id: str = "", model: str = "gpt-3.5-turbo") -> TokenCountingEvent:
    from llama_index.llms import ChatMessage

    if EventPayload.PROMPT in payload:
        prompt = str(payload.get(EventPayload.PROMPT))
        completion = str(payload.get(EventPayload.COMPLETION))

        return TokenCountingEvent(
            event_id=event_id,
            prompt=prompt,
            prompt_token_count=token_counter(model=model, text=prompt),
            completion=completion,
            completion_token_count=token_counter(model=model, text=completion),
        )

    elif EventPayload.MESSAGES in payload:
        messages = cast(List[ChatMessage], payload.get(EventPayload.MESSAGES, []))
        messages_str = "\n".join([str(x) for x in messages])
        response = str(payload.get(EventPayload.RESPONSE))

        return TokenCountingEvent(
            event_id=event_id,
            prompt=messages_str,
            prompt_token_count=token_counter(model=model, text=messages_str),
            completion=response,
            completion_token_count=token_counter(model=model, text=response),
        )
    else:
        raise ValueError("Invalid payload! Need prompt and completion or messages and response.")


def get_llm_token_costs(
    latest_llm_token_count: TokenCountingEvent,
    event_id: str = "",
    model: str = "gpt-3.5-turbo",
) -> CostCalculatingEvent:
    """
    Calculate the cost of the LLM tokens.

    Args:
        latest_llm_token_count: The latest LLM token count.
        event_id: The event id.
        model: The model to use for calculating the cost.

    Returns:
        The latest LLM token cost. (CostCalculatingEvent)
    """
    prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = cost_per_token(
        model=model,
        prompt_tokens=latest_llm_token_count.prompt_token_count,
        completion_tokens=latest_llm_token_count.completion_token_count)
    return CostCalculatingEvent(
        event_id=event_id,
        prompt=latest_llm_token_count.prompt,
        prompt_token_cost=prompt_tokens_cost_usd_dollar,
        completion=latest_llm_token_count.completion,
        completion_token_cost=completion_tokens_cost_usd_dollar,
    )


class CostCalculatingHandler(TokenCountingHandler):
    """
    Callback handler for counting costs in LLM events. (Embeddings not supported yet)

    Parameters:
        model: The model to use for tokenizer to calculate the cost.
        event_starts_to_ignore: List of event types to ignore at the start of the event.
        event_ends_to_ignore: List of event types to ignore at the end of the event.
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
        verbose: bool = False,
    ) -> None:
        self.llm_token_costs: List[CostCalculatingEvent] = []
        self.model = model

        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
            verbose=verbose,
        )

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        if (event_type == CBEventType.LLM and event_type not in self.event_ends_to_ignore and
                payload is not None):
            # token counts
            self.llm_token_counts.append(
                get_llm_token_counts(payload=payload, event_id=event_id, model=self.model))
            if self._verbose:
                print(
                    "LLM Prompt Token Usage: "
                    f"{self.llm_token_counts[-1].prompt_token_count}\n"
                    "LLM Completion Token Usage: "
                    f"{self.llm_token_counts[-1].completion_token_count}",
                    flush=True,
                )

            # token costs
            self.llm_token_costs.append(
                get_llm_token_costs(
                    latest_llm_token_count=self.llm_token_counts[-1],
                    event_id=event_id,
                    model=self.model,
                ))
            if self._verbose:
                print(
                    "LLM Total Token Cost: $"
                    f"{self.llm_token_costs[-1].total_token_cost:.6f}",
                    flush=True,
                )

        elif (event_type == CBEventType.EMBEDDING and event_type not in self.event_ends_to_ignore and
              payload is not None):
            total_chunk_tokens = 0
            for chunk in payload.get(EventPayload.CHUNKS, []):
                self.embedding_token_counts.append(
                    TokenCountingEvent(
                        event_id=event_id,
                        prompt=chunk,
                        prompt_token_count=len(self.tokenizer(chunk)),
                        completion="",
                        completion_token_count=0,
                    ))
                total_chunk_tokens += self.embedding_token_counts[-1].total_token_count

            if self._verbose:
                print(f"Embedding Token Usage: {total_chunk_tokens}", flush=True)

    @property
    def total_llm_token_cost(self) -> int:
        """Get the current total LLM token cost."""
        return sum([x.total_token_cost for x in self.llm_token_costs])

    @property
    def prompt_llm_token_cost(self) -> int:
        """Get the current total LLM prompt token cost."""
        return sum([x.prompt_token_cost for x in self.llm_token_costs])

    @property
    def completion_llm_token_cost(self) -> int:
        """Get the current total LLM completion token cost."""
        return sum([x.completion_token_cost for x in self.llm_token_costs])
