from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.readers.base import BaseReader
from llama_index.schema import Document

# Constants for content selection and cleaning
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


class WebDocsReader(BaseReader):
    """Custom reader for web documents using BeautifulSoup for parsing HTML."""

    def __init__(self, selectors: List[str] = SELECTORS, ignored_tags: List[str] = IGNORED_TAGS) -> None:
        """Initialize the reader."""
        self.selectors = selectors
        self.ignored_tags = ignored_tags
        self.visited_links = set()

    def _fetch_and_parse(self, url: str) -> BeautifulSoup:
        """Fetch the content of the web page and return a BeautifulSoup object."""
        response = requests.get(url)
        response.raise_for_status(
        )  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return BeautifulSoup(response.text, "html.parser")

    def _clean_content(self, soup: BeautifulSoup) -> str:
        """Clean the soup object to extract relevant text, ignoring the specified tags."""
        for tag in soup(self.ignored_tags):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text

    def _get_child_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Get all child links from the soup object, relative to the current_url."""
        base_url = f"{urlparse(current_url).scheme}://{urlparse(current_url).netloc}"
        return [
            urljoin(base_url, a.get('href')) for a in soup.find_all('a', href=True)
            if urlparse(a.get('href')).netloc == urlparse(base_url).netloc
        ]

    def _get_normalized_url(self, url: str) -> str:
        """Normalize the URL to avoid duplicates due to query parameters or fragments."""
        parsed_url = urlparse(url)
        return parsed_url._replace(query="", fragment="").geturl()

    def _recursive_link_collector(self, url: str, base_url: str, depth=0, max_depth=3) -> None:
        """Recursively collect links from the given URL, with depth control."""
        normalized_url = self._get_normalized_url(url)
        if normalized_url in self.visited_links or depth > max_depth:
            return
        self.visited_links.add(normalized_url)
        soup = self._fetch_and_parse(url)
        for child_url in self._get_child_links(soup, url):
            # We only follow links that lead to the same domain
            if urlparse(child_url).netloc == urlparse(base_url).netloc:
                self._recursive_link_collector(child_url, base_url, depth=depth + 1, max_depth=max_depth)

    def load_data(self, url: str) -> List[Document]:
        """Load data from the web documents starting from the given URL."""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self._recursive_link_collector(url, base_url)
        documents = []
        for url in self.visited_links:
            soup = self._fetch_and_parse(url)
            for selector in self.selectors:
                content = soup.select_one(selector)
                if content:
                    text_content = self._clean_content(content)
                    documents.append(Document(content=text_content, metadata={"url": url}))
                    break
        return documents
