from llama_index.vector_stores.qdrant import QdrantVectorStore

from autollm.utils.env_utils import read_env_variable, validate_environment_variables
from autollm.vectorstores.base import BaseVS


class QdrantVS(BaseVS):
    """Qdrant vector store."""

    def __init__(self, index_name: str, size: int = 1536, distance: str = 'EUCLID'):
        self._index_name = index_name
        self._size = size
        self._distance = distance
        self._client = None
        super().__init__()

    def _validate_requirements(self):
        """Validate all required env variables are present, and all required packages are installed."""
        required_env_variables = ['QDRANT_API_KEY', 'QDRANT_URL']

        validate_environment_variables(required_env_variables)

        try:
            import qdrant_client
        except ImportError:
            raise ImportError(
                '`qdrant-client` package not found, please run `pip install qdrant-client==1.5.4`')

    def _initialize_client(self):
        """Initialize the Qdrant client if not already initialized."""
        from qdrant_client import QdrantClient

        # If client already initialized, return
        if self._client is not None:
            return

        # Read environment variables for Qdrant initialization
        url = read_env_variable('QDRANT_URL')
        api_key = read_env_variable('QDRANT_API_KEY')

        self._client = QdrantClient(url=url, api_key=api_key)

    def initialize_vectorindex(self):
        """Create a new vector store index."""
        from qdrant_client.models import Distance, VectorParams

        # Initialize client
        self._initialize_client()

        # Convert string distance measure to Distance Enum equals to Distance.EUCLID
        distance = Distance[self._distance]

        # Create index
        self._client.recreate_collection(
            collection_name=self._index_name, vectors_config=VectorParams(size=self._size, distance=distance))

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index.

        Sets self._vectorstore.
        """
        # Initialize client
        self._initialize_client()

        # Construct vector store
        self._vectorstore = QdrantVectorStore(collection_name=self._index_name, client=self._client)
