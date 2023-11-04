import hashlib
import logging
from typing import List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.schema import Document

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


class WebDocsReader:

    def __init__(self):
        self.visited_links = set()

    def _get_child_links_recursive(self, url):
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        current_path = parsed_url.path

        response = requests.get(url)
        logging.info(f"Fetching URL: {url} - Status Code: {response.status_code}")
        if response.status_code != 200:
            logging.warning(f"Failed to fetch the website: {response.status_code}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        all_links = [link.get("href") for link in soup.find_all("a")]

        child_links = [
            link for link in all_links if link and link.startswith(current_path) and link != current_path
        ]

        absolute_paths = [urljoin(base_url, link) for link in child_links]

        logging.info(f"Child links to process: {len(absolute_paths)}")
        for link in absolute_paths:
            logging.info(f"Processing child link: {link}")
            if link not in self.visited_links:
                self.visited_links.add(link)
                self._get_child_links_recursive(link)

    def _get_all_urls(self, url):
        self.visited_links = set()
        self._get_child_links_recursive(url)
        urls = [link for link in self.visited_links if urlparse(link).netloc == urlparse(url).netloc]
        return urls

    def _load_data_from_url(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            logging.info(f"Failed to fetch the website: {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        output = []
        for selector in SELECTORS:
            element = soup.select_one(selector)
            if element:
                logging.info(f"Found element with selector: {selector} for URL: {url}")
                content = element.prettify()
                break
        else:
            logging.info(f"Failed to find any element for URL: {url}")
            content = soup.get_text()

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(IGNORED_TAGS):
            tag.decompose()

        content = " ".join(soup.stripped_strings)
        output.append({
            "content": content,
            "meta_data": {
                "url": url
            },
        })

        return output

    def load_data(self, url: str) -> List[Document]:
        all_urls = self._get_all_urls(url)
        logging.info(f"Total URLs to process: {len(all_urls)}")
        output = []
        for u in all_urls:
            output.extend(self._load_data_from_url(u))

        documents = []
        for d in output:
            document = Document(text=d['content'], metadata=d['meta_data'])
            documents.append(document)
        logging.info(f"Total documents created: {len(documents)}")
        return documents
