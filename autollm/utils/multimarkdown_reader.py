from pathlib import Path
from typing import Dict, List, Optional

from llama_index.schema import Document

from autollm.utils.hash_utils import get_md5
from autollm.utils.markdown_reader import MarkdownReader


class MultiMarkdownReader(MarkdownReader):
    """
    MultiMarkdown parser.

    Extract text from multiple markdown files. Returns a list of dictionaries with keys as headers and values
    as the text between headers.
    """

    def __init__(self, *args, read_as_single_doc: bool = False, **kwargs) -> None:
        """
        Initialize MultiMarkdownReader.

        Parameters:
            read_as_single_doc (bool): If True, read each markdown as a single document.
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

        relative_file_path = str(file)
        # TODO: check if this is still necessary after llama_index refresh method entegration
        extra_info['original_file_path'] = relative_file_path
        extra_info['md5_hash'] = get_md5(file)

        if self.read_as_single_doc:
            # Reading entire markdown as a single document
            with open(file, encoding='utf-8') as f:
                content = f.read()
            if self._remove_hyperlinks:
                content = self.remove_hyperlinks(content)
            if self._remove_images:
                content = self.remove_images(content)
            # Generate doc_id as file name
            doc_id = relative_file_path

            return [Document(id_=doc_id, text=content, metadata=extra_info)]
        else:
            # Call parent's load_data method for section-based reading
            return super().load_data(file, extra_info, content)
