"""LanceDB vector store with cloud storage support."""
from typing import Any, Optional

from llama_index.vector_stores import LanceDBVectorStore as LanceDBVectorStoreBase


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
