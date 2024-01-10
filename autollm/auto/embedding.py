from typing import Any

from litellm import aembedding as lite_aembedding
from litellm import embedding as lite_embedding
from llama_index.bridge.pydantic import Field
from llama_index.embeddings.base import BaseEmbedding, Embedding


class AutoEmbedding(BaseEmbedding):
    """
    Custom embedding class for flexible and efficient text embedding.

    This class interfaces with the LiteLLM library to use its embedding functionality, making it compatible
    with a wide range of LLM models.
    """

    # Define the model attribute using Pydantic's Field
    model: str = Field(default="text-embedding-ada-002", description="The name of the embedding model.")

    def __init__(self, model: str, **kwargs: Any) -> None:
        """
        Initialize the AutoEmbedding with a specific model.

        Args:
            model (str): ID of the embedding model to use.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model = model  # Set the model ID for embedding

    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Synchronously get the embedding for a query string.

        Args:
            query (str): The query text to embed.

        Returns:
            Embedding: The embedding vector.
        """
        response = lite_embedding(model=self.model, input=[query])
        return self._parse_embedding_response(response)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Asynchronously get the embedding for a query string.

        Args:
            query (str): The query text to embed.

        Returns:
            Embedding: The embedding vector.
        """
        response = await lite_aembedding(model=self.model, input=[query])
        return self._parse_embedding_response(response)

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Synchronously get the embedding for a text string.

        Args:
            text (str): The text to embed.

        Returns:
            Embedding: The embedding vector.
        """
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """
        Asynchronously get the embedding for a text string.

        Args:
            text (str): The text to embed.

        Returns:
            Embedding: The embedding vector.
        """
        return await self._aget_query_embedding(text)

    def _parse_embedding_response(self, response):
        """
        Parse the embedding response from LiteLLM and extract the embedding data.

        Args:
            response: The response object from LiteLLM's embedding function.

        Returns:
            List[float]: The extracted embedding list.
        """
        try:
            if 'data' in response and len(response['data']) > 0 and 'embedding' in response['data'][0]:
                return response['data'][0]['embedding']
            else:
                raise ValueError("Invalid response structure from embedding function.")
        except (TypeError, KeyError, IndexError) as e:
            # Handle any parsing errors
            raise ValueError(f"Error parsing embedding response: {e}")
