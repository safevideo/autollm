"""LanceDB vector store with cloud storage support."""
import os
from typing import Any, Optional

from dotenv import load_dotenv
from llama_index.vector_stores import LanceDBVectorStore as LanceDBVectorStoreBase

load_dotenv()


class LanceDBVectorStore(LanceDBVectorStoreBase):

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
        import_err_msg = "`lancedb` package not found, please run `pip install lancedb`"
        try:
            import lancedb
        except ImportError:
            raise ImportError(import_err_msg)

        # Check for API key and region in environment variables if not provided
        if api_key is None:
            api_key = os.getenv('LANCEDB_API_KEY')
        if region is None:
            region = os.getenv('LANCEDB_REGION')

        if api_key and region:
            self.connection = lancedb.connect(uri, api_key=api_key, region=region)
        else:
            self.connection = lancedb.connect(uri)

        self.uri = uri
        self.table_name = table_name
        self.nprobes = nprobes
        self.refine_factor = refine_factor
        self.api_key = api_key
        self.region = region
