import os
import shutil
from typing import Optional, Sequence

from llama_index import Document, ServiceContext, StorageContext, VectorStoreIndex
from llama_index.schema import BaseNode

from autollm.utils.env_utils import on_rm_error
from autollm.utils.lancedb_vectorstore import LanceDBVectorStore
from autollm.utils.logging import logger


def import_vector_store_class(vector_store_class_name: str):
    """
    Imports a predefined vector store class by class name.

    Args:
    Returns:
        The imported VectorStore class.
    """
    module = __import__("llama_index.vector_stores", fromlist=[vector_store_class_name])
    class_ = getattr(module, vector_store_class_name)
    return class_


class AutoVectorStoreIndex:
    """AutoVectorStoreIndex lets you dynamically initialize any Vector Store index based on the vector store
    class name and additional parameters.
    """

    @staticmethod
    def from_defaults(
            vector_store_type: str = "LanceDBVectorStore",
            lancedb_uri: str = None,
            lancedb_table_name: str = "vectors",
            lancedb_api_key: Optional[str] = None,
            lancedb_region: Optional[str] = None,
            documents: Optional[Sequence[Document]] = None,
            nodes: Optional[Sequence[BaseNode]] = None,
            service_context: Optional[ServiceContext] = None,
            exist_ok: bool = False,
            overwrite_existing: bool = False,
            **kwargs) -> VectorStoreIndex:
        """
        Initializes a Vector Store index from Vector Store type and additional parameters. Handles lancedb
        path and document management according to specified behaviors.

        Parameters:
            vector_store_type (str): The class name of the vector store.
            lancedb_uri (str): The URI for the LanceDB vector store.
            lancedb_table_name (str): The table name for the LanceDB vector store.
            documents (Optional[Sequence[Document]]): Documents to initialize the vector store index from.
            service_context (Optional[ServiceContext]): Service context for initialization.
            exist_ok (bool): If True, allows adding to an existing database.
            overwrite_existing (bool): If True, allows overwriting an existing database.
            **kwargs: Additional parameters for initialization.

        Returns:
            VectorStoreIndex: The initialized Vector Store index instance.

        Raises:
            ValueError: For invalid parameter combinations or missing information.
        """
        if documents is None and nodes is None and vector_store_type == "SimpleVectorStore":
            raise ValueError("documents or nodes must be provided for SimpleVectorStore")

        if documents is not None and nodes is not None:
            raise ValueError("documents and nodes cannot be provided at the same time")

        # Initialize vector store
        VectorStoreClass = import_vector_store_class(vector_store_type)

        # If LanceDBVectorStore, use lancedb_uri and lancedb_table_name
        if vector_store_type == "LanceDBVectorStore":
            lancedb_uri = AutoVectorStoreIndex._validate_and_setup_lancedb_uri(
                lancedb_uri=lancedb_uri,
                documents=documents,
                exist_ok=exist_ok,
                overwrite_existing=overwrite_existing)

            vector_store = LanceDBVectorStore(
                uri=lancedb_uri,
                table_name=lancedb_table_name,
                api_key=lancedb_api_key,
                region=lancedb_region,
                **kwargs)

        else:
            vector_store = VectorStoreClass(**kwargs)

        # Initialize vector store index from existing vector store
        if documents is None and nodes is None:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, service_context=service_context)
            return index

        # Initialize vector store index from documents or nodes
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if documents is not None:
            index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=True)
        else:
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                service_context=service_context,
                show_progress=True)

        return index

    @staticmethod
    def _validate_and_setup_lancedb_uri(lancedb_uri, documents, exist_ok, overwrite_existing):
        """
        Validates and sets up the lancedb_uri based on the given parameters.

        Parameters:
            lancedb_uri (str): The URI for the LanceDB vector store.
            documents (Sequence[Document]): Documents to initialize the vector store index from.
            exist_ok (bool): Flag to allow adding to an existing database.
            overwrite_existing (bool): Flag to allow overwriting an existing database.

        Returns:
            str: The validated and potentially modified lancedb_uri.
        """
        default_lancedb_uri = "./lancedb/db"

        # Scenario 0: Handle no lancedb uri and no documents provided
        if not documents and not lancedb_uri:
            raise ValueError(
                "A lancedb uri is required to connect to a database. Please provide a lancedb uri.")

        # Scenario 1: Handle lancedb_uri given but no documents provided
        if not documents and lancedb_uri:
            # Check if the database exists
            db_exists = os.path.exists(lancedb_uri)
            if not db_exists:
                raise ValueError(
                    f"No existing database found at {lancedb_uri}. Please provide a valid lancedb uri.")

        # Scenario 2: Handle no lancedb uri but documents provided
        if documents and not lancedb_uri:
            lancedb_uri = default_lancedb_uri
            lancedb_uri = AutoVectorStoreIndex._increment_lancedb_uri(lancedb_uri)
            logger.info(
                f"A new database is being created at {lancedb_uri}. Please provide a lancedb path to use an existing database."
            )

        # Scenario 3: Handle lancedb uri given and documents provided
        if documents and lancedb_uri:
            db_exists = os.path.exists(lancedb_uri)
            if exist_ok and overwrite_existing:
                if db_exists:
                    shutil.rmtree(lancedb_uri)
                    logger.info(f"Overwriting existing database at {lancedb_uri}.")
            elif not exist_ok and overwrite_existing:
                raise ValueError("Cannot overwrite existing database without exist_ok set to True.")
            elif db_exists:
                if not exist_ok:
                    lancedb_uri = AutoVectorStoreIndex._increment_lancedb_uri(lancedb_uri)
                    logger.info(f"Existing database found. Creating a new database at {lancedb_uri}.")
                    logger.info(
                        "Please use exist_ok=True to add to the existing database and overwrite_existing=True to overwrite the existing database."
                    )
                else:
                    logger.info(f"Adding documents to existing database at {lancedb_uri}.")

        return lancedb_uri

    @staticmethod
    def _increment_lancedb_uri(base_uri: str) -> str:
        """Increment the lancedb uri to create a new database."""
        i = 1
        while os.path.exists(f"{base_uri}_{i}"):
            i += 1
        return f"{base_uri}_{i}"
