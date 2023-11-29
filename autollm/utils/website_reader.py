import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.schema import Document
from tqdm import tqdm

from autollm.utils.constants import WEBPAGE_READER_TIMEOUT
from autollm.utils.logging import logger
from autollm.utils.webpage_reader import WebPageReader


class WebSiteReader:

    def __init__(self):
        self.visited_links = set()

    def _fetch_and_parse_sitemap(
            self,
            sitemap_url: str,
            include_filter_str: Optional[str] = None,
            exclude_filter_str: Optional[str] = None) -> List[str]:
        """Fetches and parses the sitemap, returning URLs."""
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()
            sitemap_content = response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching sitemap: {e}")
            return []

        return self._extract_urls_from_sitemap(sitemap_content, include_filter_str, exclude_filter_str)

    def _extract_urls_from_sitemap(
            self, sitemap_content: str, include_filter_str: Optional[str],
            exclude_filter_str: Optional[str]) -> List[str]:
        """Extracts URLs from sitemap content."""
        sitemap = ET.fromstring(sitemap_content)
        urls = []

        for url in sitemap.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            location = url.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc").text
            if (include_filter_str and
                    include_filter_str not in location) or (exclude_filter_str and
                                                            exclude_filter_str in location):
                continue
            urls.append(location)

        return urls

    def _get_child_links_recursive(self, url):
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        current_path = parsed_url.path

        response = requests.get(url, timeout=WEBPAGE_READER_TIMEOUT)  # timeout in seconds
        if response.status_code != 200:
            logger.warning(f"Failed to fetch the website: {response.status_code}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        all_links = [link.get("href") for link in soup.find_all("a")]

        # Normalize links and filter out external links and anchors
        child_links = set()
        for link in all_links:
            # Skip any None or empty hrefs, and anchors
            if not link or link.startswith('#') or link == current_path:
                continue
            # Convert relative links to absolute
            full_link = urljoin(base_url, link)
            # Add to set if the link is internal
            if urlparse(full_link).netloc == parsed_url.netloc:
                child_links.add(full_link)

        # Process each child link
        for link in child_links:
            if link not in self.visited_links:
                self.visited_links.add(link)
                self._get_child_links_recursive(link)

    def _get_all_urls(self, url):
        self.visited_links = set()
        self._get_child_links_recursive(url)
        urls = [link for link in self.visited_links if urlparse(link).netloc == urlparse(url).netloc]
        return urls

    def load_data(
            self,
            parent_url: str = None,
            sitemap_url: str = None,
            include_filter_str: str = None,
            exclude_filter_str: str = None) -> List[Document]:
        """Loads data from either a standard URL or a sitemap URL."""
        if sitemap_url:
            logger.info(f"Fetching and parsing sitemap {sitemap_url}..")
            all_urls = self._fetch_and_parse_sitemap(sitemap_url, include_filter_str, exclude_filter_str)
        elif parent_url:
            logger.info(f"Parsing child pages of {parent_url}..")
            all_urls = self._get_child_links_recursive(parent_url)
        else:
            raise ValueError("Either sitemap_url or parent_url must be provided.")

        web_reader = WebPageReader()
        documents = []

        logger.info(f"Total URLs to process: {len(all_urls)}")

        for u in tqdm(all_urls, desc="Processing URLs"):
            if u not in self.visited_links:
                self.visited_links.add(u)
                documents.extend(web_reader.load_data(u))

        return documents
