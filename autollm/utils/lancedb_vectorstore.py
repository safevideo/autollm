"""LanceDB vector store with cloud storage support."""
import os
from typing import Any, Optional

from dotenv import load_dotenv
from llama_index.vector_stores import LanceDBVectorStore as LanceDBVectorStoreBase

load_dotenv()


class LanceDBVectorStore(LanceDBVectorStoreBase):
    """Advanced LanceDB Vector Store supporting cloud storage and prefiltering."""

    def __init__(
        self,
        uri: str,
        table_name: str = "vectors",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self._setup_connection(uri, api_key, region)
        self.uri = uri
        self.table_name = table_name
        self.nprobes = nprobes
        self.refine_factor = refine_factor
        self.api_key = api_key
        self.region = region

    def _setup_connection(self, uri: str, api_key: Optional[str], region: Optional[str]):
        """Establishes a robust connection to LanceDB."""
        api_key = api_key or os.getenv('LANCEDB_API_KEY')
        region = region or os.getenv('LANCEDB_REGION')

        import_err_msg = "`lancedb` package not found, please run `pip install lancedb`"
        try:
            import lancedb
        except ImportError:
            raise ImportError(import_err_msg)

        if api_key and region:
            self.connection = lancedb.connect(uri, api_key=api_key, region=region)
        else:
            self.connection = lancedb.connect(uri)
