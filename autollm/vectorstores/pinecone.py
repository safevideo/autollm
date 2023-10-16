from llama_index.vector_stores import PineconeVectorStore

from autollm.utils.env_utils import read_env_variable, validate_environment_variables
from autollm.vectorstores.base import BaseVS


class PineconeVS(BaseVS):

    def __init__(
            self, index_name: str, dimension: int = 1536, metric: str = 'euclidean', pod_type: str = 'p1'):
        self._index_name = index_name
        self._dimension = dimension
        self._metric = metric
        self._pod_type = pod_type
        super().__init__()

    def _validate_requirements(self):
        """Validate all required env variables are present, and all required packages are installed."""
        required_env_vars = ['PINECONE_API_KEY', 'PINECONE_ENVIRONMENT']

        validate_environment_variables(required_env_vars)

        try:
            import pinecone
        except ImportError:
            raise ImportError('`pinecone` package not found, please run `pip install pinecone-client==2.2.4`')

    def initialize_vectorindex(self):
        """Create a new vector store index."""
        import pinecone

        # Read environment variables for Pinecone initialization
        api_key = read_env_variable('PINECONE_API_KEY')
        environment = read_env_variable('PINECONE_ENVIRONMENT')

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        # Dimensions are for text-embedding-ada-002
        pinecone.create_index(
            self._index_name, dimension=self._dimension, metric=self._metric, pod_type=self._pod_type)

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index.

        Sets self._vectorstore.
        """
        import pinecone

        index = pinecone.Index(self._index_name)
        # Construct vector store
        self._vectorstore = PineconeVectorStore(pinecone_index=index)
