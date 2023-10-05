from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from llama_index.schema import Document

from markdown_reader import MarkdownReader


class MultiMarkdownReader(MarkdownReader):
    """MultiMarkdown parser.
    Extract text from multiple markdown files.
    Returns a list of dictionaries with keys as headers and values as the text between headers.
    """

    def __init__(self, *args, read_as_single_doc: bool = False, **kwargs) -> None:
        """Initialize MultiMarkdownReader.
        
        Parameters:
            read_as_single_doc (bool): Flag to read entire markdown as a single document.
        """
        super().__init__(*args, **kwargs)
        self.read_as_single_doc = read_as_single_doc

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        content: Optional[str] = None,
    ) -> List[Document]:
        """Include original_file_path in extra_info and respect read_as_single_doc flag."""
        if extra_info is None:
            extra_info = {}
        
        extra_info["original_file_path"] = str(file)
        
        if self.read_as_single_doc:
            # Reading entire markdown as a single document
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Generate doc_id as file name
            doc_id = str(file)
            
            return [
                Document(
                    id_=doc_id,
                    text=content,
                    metadata=extra_info
                )
            ]
        else:
            # Call parent's load_data method for section-based reading
            return super().load_data(file, extra_info, content)
    
    def load_data_from_folder(
            self,
            folder_path: Path,
            extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Parse all markdown files in a given folder into Documents."""
        all_documents = []
        
        # Gather all markdown files in the folder and its subfolders
        all_files = list(folder_path.rglob("*.md"))
        
        for file_path in all_files:
            # Use the overridden load_data method
            documents = self.load_data(file_path, extra_info)
            all_documents.extend(documents)

        return all_documents
