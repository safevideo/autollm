"""LanceDB vector store with cloud storage support."""
import os
from typing import Any, Optional

from dotenv import load_dotenv
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import LanceDBVectorStore as LanceDBVectorStoreBase
from llama_index.vector_stores.lancedb import _to_lance_filter, _to_llama_similarities
from llama_index.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from pandas import DataFrame

load_dotenv()


class LanceDBVectorStore(LanceDBVectorStoreBase):
    """Advanced LanceDB Vector Store supporting cloud storage and prefiltering."""
    from lancedb.query import LanceQueryBuilder
    from lancedb.table import Table

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

    def _setup_connection(self, uri: str, api_key: Optional[str] = None, region: Optional[str] = None):
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

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Enhanced query method to support prefiltering in LanceDB queries."""
        table = self.connection.open_table(self.table_name)
        lance_query = self._prepare_lance_query(query, table, **kwargs)

        results = lance_query.to_df()
        return self._construct_query_result(results)

    def _prepare_lance_query(self, query: VectorStoreQuery, table: Table, **kwargs) -> LanceQueryBuilder:
        """Prepares the LanceDB query considering prefiltering and additional parameters."""
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError(
                    "Cannot specify filter via both query and kwargs. "
                    "Use kwargs only for lancedb specific items that are "
                    "not supported via the generic query interface.")
            where = _to_lance_filter(query.filters)
        else:
            where = kwargs.pop("where", None)
        prefilter = kwargs.pop("prefilter", False)

        table = self.connection.open_table(self.table_name)
        lance_query = (
            table.search(query.query_embedding).limit(query.similarity_top_k).where(
                where, prefilter=prefilter).nprobes(self.nprobes))

        if self.refine_factor is not None:
            lance_query.refine_factor(self.refine_factor)

        return lance_query

    def _construct_query_result(self, results: DataFrame) -> VectorStoreQueryResult:
        """Constructs a VectorStoreQueryResult from a LanceDB query result."""
        nodes = []

        for _, row in results.iterrows():
            node = TextNode(
                text=row.get('text', ''),  # ensure text is a string
                id_=row['id'],
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=row['doc_id']),
                })
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=_to_llama_similarities(results),
            ids=results["id"].tolist(),
        )
