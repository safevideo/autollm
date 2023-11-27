from typing import List

import requests
from bs4 import BeautifulSoup
from llama_index.schema import Document

from autollm.utils.constants import WEBPAGE_READER_TIMEOUT
from autollm.utils.logging import logger

# Constants defined outside the class
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
    """A class for reading and processing the content of a single web page."""

    def load_data(self, url: str) -> List[Document]:
        """
        Reads a web page from the provided URL, extracts content using predefined selectors, removes all
        ignored tags, and returns a list of Document objects with the processed content.

        Parameters:
            url (str): The URL of the web page to read.

        Returns:
            List[Document]: A list containing Document objects with the processed content and its metadata.
        """
        response = requests.get(url, timeout=WEBPAGE_READER_TIMEOUT)
        if response.status_code != 200:
            logger.info(f"Failed to fetch the website: {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        for selector in SELECTORS:
            element = soup.select_one(selector)
            if element:
                content = element.prettify()
                break
        else:
            logger.info(f"Failed to find any element for URL: {url}")
            content = soup.get_text()

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(IGNORED_TAGS):
            tag.decompose()

        content = " ".join(soup.stripped_strings)
        document = Document(id_=url, text=content, metadata={"url": url})
        logger.info(f"Processed URL: {url}")

        return [document]


# Example of usage:
# reader = WebPageReader()
# documents = reader.load_data('http://example.com/some-page')
