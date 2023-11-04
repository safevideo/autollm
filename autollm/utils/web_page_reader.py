from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.schema import Document

# Selectors to identify the main content of the page
SELECTORS = [
    "article.bd-article",
    'article[role="main"]',
    "div.md-content",
    'div[role="main"]',
    "div.container",
    "div.section",
    "article",
    "main",
]

# Tags to ignore while extracting content
IGNORED_TAGS = [
    "nav",
    "aside",
    "form",
    "header",
    "noscript",
    "svg",
    "canvas",
    "footer",
    "script",
    "style",
]


class WebPageReader:
    """Class to read and process content from a single web page."""

    def load_data(self, url: str) -> List[Document]:
        """
        Load and process content from the given URL.

        Args:
            url (str): The URL of the web page to process.

        Returns:
            List[Document]: A list containing a single Document object with the processed content.
        """
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch the website: {response.status_code}")

        soup = BeautifulSoup(response.content, "html.parser")

        content = None
        for selector in SELECTORS:
            element = soup.select_one(selector)
            if element:
                content = element.prettify()
                break
        if content is None:
            content = soup.get_text()

        # Clean the content by removing ignored tags
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(IGNORED_TAGS):
            tag.decompose()

        content_cleaned = " ".join(soup.stripped_strings)
        document = Document(text=content_cleaned, metadata={"url": url})

        return [document]
