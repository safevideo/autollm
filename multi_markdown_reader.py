from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from llama_index.readers.base import BaseReader
from llama_index.schema import Document
import re
import os

class MultiMarkdownReader(BaseReader):
    """MultiMarkdown parser.
    Extract text from multiple markdown files.
    Returns a list of dictionaries with keys as headers and values as the text between headers.
    """

    def __init__(
        self,
        *args: Any,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        """Convert a markdown file to a list of tuples.
        The first element in each tuple is the header and the second is the text under the header.
        """
        markdown_tups = []
        lines = markdown_text.split("\n")
        current_header = None
        current_text = ""

        for line in lines:
            header_match = re.match(r"^#+\s", line)
            if header_match:
                if current_header is not None:
                    markdown_tups.append((current_header, current_text))
                current_header = line
                current_text = ""
            else:
                current_text += line + "\n"
                
        if current_header is not None:
            markdown_tups.append((current_header, current_text))
            
        return markdown_tups

    def remove_images(self, content: str) -> str:
        """Remove image links."""
        pattern = r"!\[(.*?)\]\((.*?)\)"
        return re.sub(pattern, "", content)

    def remove_hyperlinks(self, content: str) -> str:
        """Remove hyperlinks."""
        pattern = r"\[(.*?)\]\((.*?)\)"
        return re.sub(pattern, r"\1", content)

    def load_data_from_folder(self, folder_path: Path, extra_info: Optional[Dict] = None) -> List[Document]:
        """Parse all markdown files in a given folder into Documents."""
        all_documents = []
        
        # Loop through each markdown file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".md"):
                file_path = folder_path / file_name
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                if self._remove_hyperlinks:
                    content = self.remove_hyperlinks(content)
                if self._remove_images:
                    content = self.remove_images(content)
                    
                markdown_tups = self.markdown_to_tups(content)
                
                for header, value in markdown_tups:
                    document = Document(
                        text=f"\n\n{header}\n{value}" if header else value,
                        metadata=extra_info or {}
                    )
                    all_documents.append(document)
                    
        return all_documents

