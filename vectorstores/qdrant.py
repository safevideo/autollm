from llama_index.vector_stores.qdrant import QdrantVectorStore
from utils.env_utils import read_env_variable, validate_environment_variables
from .base import BaseVS

class QdrantVS(BaseVS):
    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self._client = None
        self._url = None
        self._api_key = None
        super().__init__()

    def _validate_requirements(self):
        """
        Validate all required env variables are present, and all required packages are installed.
        """
        required_env_variables = ["QDRANT_API_KEY", "QDRANT_URL"]

        validate_environment_variables(required_env_variables)

        try:
            import qdrant_client
        except ImportError:
            raise ImportError("`qdrant-client` package not found, please run `pip install qdrant-client`")

    def _initialize_client(self):
        """
        Initialize the Qdrant client if not already initialized.
        """
        import qdrant_client

        # If client already initialized, return
        if self._client is not None:
            return
        
        # Read environment variables for Qdrant initialization
        self._url = read_env_variable("QDRANT_URL")
        self._api_key = read_env_variable("QDRANT_API_KEY")
        
        self._client = qdrant_client.QdrantClient(
            url=self._url, 
            api_key=self._api_key, 
            **self._client_kwargs
        )

    def initialize_vectorindex(self):
        """
        Create a new vector store index.
        """
        # Initialize client
        self._initialize_client()
        
        # Create index
        self._client.create_collection(
            self._collection_name,
            vectors_config={
                "dimension": self._dimension,
                "distance": self._metric
            }
        )

    def connect_vectorstore(self):
        """
        Connect to an existing vector store index. Sets self._vectorstore.
        """
        # Initialize client
        self._initialize_client()

        # Construct vector store
        self._vectorstore = QdrantVectorStore(
            collection_name=self._collection_name,
            client=self._client
        )
