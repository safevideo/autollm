from typing import List

from llama_index.readers.base import BaseReader
from llama_index.schema import Document

from autollm.utils.logging import logger


class LangchainPDFReader(BaseReader):
    """Custom PDF reader that uses langchain's PDFMinerLoader."""

    def __init__(self, extract_images: bool = False) -> None:
        """Initialize the reader."""
        self.extract_images = extract_images

    def load_data(self, file_path: str, extra_info: dict = None) -> List[Document]:
        """Load data from a PDF file using langchain's PDFMinerLoader."""
        from langchain.document_loaders import PDFMinerLoader

        # Convert the PosixPath object to a string before passing it to PDFMinerLoader
        loader = PDFMinerLoader(str(file_path), extract_images=self.extract_images)

        logger.info(f"Parsing pages of the PDF file: {file_path}..")
        langchain_documents = loader.load()  # This returns a list of langchain Document objects

        # Convert langchain documents into llama-index documents
        documents = []
        for langchain_document in langchain_documents:
            # Create a llama-index document for each langchain document
            doc = Document.from_langchain_format(langchain_document)

            # If there's extra info, we can add it to the Document's metadata
            if extra_info is not None:
                doc.metadata.update(extra_info)

            documents.append(doc)

        return documents
