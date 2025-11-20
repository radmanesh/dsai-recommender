"""Website crawler for faculty websites and lab websites."""

import time
import re
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from src.utils.config import Config
from src.utils.logger import get_logger, debug, info, warning, error, verbose

logger = get_logger(__name__)


def extract_urls_from_csv_docs(csv_docs: List[Document]) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract Website and Lab Website URLs from CSV documents.

    Args:
        csv_docs: List of CSV documents with faculty metadata.

    Returns:
        Dict mapping faculty_id to list of URL dicts with 'url' and 'source_type' keys.
    """
    url_mapping = {}

    for doc in csv_docs:
        faculty_id = doc.metadata.get("faculty_id", "")
        faculty_name = doc.metadata.get("faculty_name", "Unknown")

        if not faculty_id:
            continue

        urls = []

        # Extract Website URL
        website = doc.metadata.get("website") or doc.metadata.get("Website")
        if website and website.strip() and website.strip().lower() not in ["", "n/a", "none"]:
            url = website.strip()
            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            urls.append({
                "url": url,
                "source_type": "faculty_website"
            })

        # Extract Lab Website URL
        lab_website = doc.metadata.get("lab_website") or doc.metadata.get("Lab Website")
        if lab_website and lab_website.strip() and lab_website.strip().lower() not in ["", "n/a", "none"]:
            url = lab_website.strip()
            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            urls.append({
                "url": url,
                "source_type": "lab_website"
            })

        if urls:
            url_mapping[faculty_id] = {
                "faculty_id": faculty_id,
                "faculty_name": faculty_name,
                "urls": urls
            }

    return url_mapping


def _normalize_url(url: str, base_url: str) -> str:
    """Normalize a URL relative to a base URL."""
    if url.startswith(("http://", "https://")):
        return url
    return urljoin(base_url, url)


def _get_directory_path(url: str) -> str:
    """
    Get the directory path from a URL.

    Examples:
    - https://ou.edu/cashbarker -> /cashbarker/
    - https://ou.edu/cashbarker/index.html -> /cashbarker/
    - https://ou.edu/cashbarker/ -> /cashbarker/

    Args:
        url: The URL to parse.

    Returns:
        The directory path (always ends with /), or empty string if root.
    """
    try:
        parsed = urlparse(url)
        path = parsed.path

        # Remove filename if it's a file (has extension)
        if '.' in path.split('/')[-1] and path.split('/')[-1] != '.' and path.split('/')[-1] != '..':
            # It's likely a file, get the directory
            path = '/'.join(path.split('/')[:-1])

        # Ensure path ends with / (unless it's root)
        if path and not path.endswith('/'):
            path += '/'

        # If path is just /, return empty string (root level)
        if path == '/':
            return ''

        return path
    except Exception:
        return ''


def _is_same_directory(url: str, base_directory_path: str, base_domain: str) -> bool:
    """
    Check if a URL is in the same directory path as the base URL.

    Args:
        url: The URL to check.
        base_directory_path: The directory path from the base URL (from _get_directory_path).
        base_domain: The domain from the base URL.

    Returns:
        True if URL is in the same directory path, False otherwise.
    """
    try:
        parsed = urlparse(url)
        link_domain = parsed.netloc.lower()
        link_path = parsed.path

        base_domain_lower = base_domain.lower()

        # Remove www. prefix for comparison
        if link_domain.startswith("www."):
            link_domain = link_domain[4:]
        if base_domain_lower.startswith("www."):
            base_domain_lower = base_domain_lower[4:]

        # First check if it's the same domain
        same_domain = (link_domain == base_domain_lower or
                      link_domain == "" or
                      link_domain.endswith("." + base_domain_lower))

        if not same_domain:
            return False

        # Normalize link path (remove filename if it's a file)
        link_directory = _get_directory_path(url)

        # If base_directory_path is empty (root), only accept root level
        if not base_directory_path:
            return link_directory == '' or link_directory == '/'

        # Ensure both paths end with / for consistent comparison
        base_path_normalized = base_directory_path if base_directory_path.endswith('/') else base_directory_path + '/'
        link_directory_normalized = link_directory if link_directory.endswith('/') else link_directory + '/'

        # Check if link directory starts with base directory path
        return link_directory_normalized.startswith(base_path_normalized)

    except Exception:
        return False


def _extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)

    return text


def crawl_website(
    base_url: str,
    max_pages: int = 20,
    max_depth: int = 3,
    timeout: int = 30,
    rate_limit_delay: float = 1.0
) -> List[Dict[str, str]]:
    """
    Crawl a single website and return all internal pages.

    Args:
        base_url: Starting URL to crawl.
        max_pages: Maximum number of pages to crawl.
        max_depth: Maximum depth to crawl.
        timeout: Request timeout in seconds.
        rate_limit_delay: Delay between requests in seconds.

    Returns:
        List of dicts with 'url', 'title', and 'content' keys.
    """
    if not base_url or not base_url.strip():
        return []

    try:
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        base_directory_path = _get_directory_path(base_url)
        debug(f"Parsing base URL: {base_url} -> domain: {base_domain}, directory: {base_directory_path}")
    except Exception as e:
        warning(f"Invalid URL {base_url}: {e}")
        verbose(f"URL parsing exception: {type(e).__name__}: {str(e)}")
        return []

    visited: Set[str] = set()
    to_visit: List[tuple[str, int]] = [(base_url, 0)]  # (url, depth)
    pages: List[Dict[str, str]] = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    debug(f"Starting crawl: url={base_url}, max_pages={max_pages}, max_depth={max_depth}, timeout={timeout}s")
    verbose(f"Crawl configuration: base_domain={base_domain}, base_directory={base_directory_path}, rate_limit={rate_limit_delay}s")

    while to_visit and len(pages) < max_pages:
        current_url, depth = to_visit.pop(0)

        # Skip if already visited or too deep
        if current_url in visited or depth > max_depth:
            if current_url in visited:
                verbose(f"Skipping already visited URL: {current_url}")
            if depth > max_depth:
                verbose(f"Skipping URL at depth {depth} (max {max_depth}): {current_url}")
            continue

        visited.add(current_url)
        debug(f"Crawling page {len(pages) + 1}/{max_pages}: {current_url} (depth: {depth})")

        try:
            # Rate limiting
            if len(pages) > 0:
                verbose(f"Rate limiting: waiting {rate_limit_delay}s...")
                time.sleep(rate_limit_delay)

            # Fetch page
            verbose(f"Fetching: {current_url}")
            response = requests.get(current_url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            debug(f"HTTP {response.status_code} - {len(response.content)} bytes")

            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                verbose(f"Skipping non-HTML content: {content_type}")
                continue

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            verbose(f"HTML parsed: {len(soup.get_text())} characters of text")

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ""
            debug(f"Page title: {title[:60]}...")

            # Extract text content
            text_content = _extract_text_from_html(str(soup))
            verbose(f"Extracted text: {len(text_content)} characters")

            # Skip if content is too short (likely not useful)
            if len(text_content) < 100:
                debug(f"Skipping short content: {len(text_content)} chars < 100")
                continue

            pages.append({
                "url": current_url,
                "title": title,
                "content": text_content
            })
            debug(f"Page {len(pages)} added: {title[:40]}...")

            # Extract links for further crawling
            if depth < max_depth:
                links_found = soup.find_all('a', href=True)
                debug(f"Found {len(links_found)} links on page")
                internal_links_added = 0

                for link in links_found:
                    href = link['href']
                    absolute_url = _normalize_url(href, current_url)

                    # Only follow links in the same directory path
                    if _is_same_directory(absolute_url, base_directory_path, base_domain) and absolute_url not in visited:
                        # Avoid common non-content URLs
                        if not any(skip in absolute_url.lower() for skip in [
                            '#', 'mailto:', 'tel:', 'javascript:', '.pdf', '.doc', '.docx',
                            '.zip', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js'
                        ]):
                            to_visit.append((absolute_url, depth + 1))
                            internal_links_added += 1
                            verbose(f"Added link in same directory to queue: {absolute_url}")
                        else:
                            verbose(f"Skipping non-content URL: {absolute_url}")
                    else:
                        if absolute_url in visited:
                            verbose(f"Skipping already visited URL: {absolute_url}")
                        else:
                            verbose(f"Skipping URL outside directory scope: {absolute_url} (base directory: {base_directory_path})")

                debug(f"Added {internal_links_added} links in same directory to crawl queue")

        except requests.exceptions.RequestException as e:
            warning(f"Error fetching {current_url}: {e}")
            verbose(f"Request exception: {type(e).__name__}: {str(e)}")
            continue
        except Exception as e:
            warning(f"Error processing {current_url}: {e}")
            verbose(f"Processing exception: {type(e).__name__}: {str(e)}")
            continue

    debug(f"Crawl complete: {len(pages)} pages from {base_url}")
    verbose(f"Visited URLs: {len(visited)}, Pages collected: {len(pages)}")

    return pages


def crawl_faculty_websites(
    csv_docs: List[Document],
    max_pages_per_site: int = 20,
    max_depth: int = 3,
    timeout: int = 30,
    rate_limit_delay: float = 1.0
) -> List[Document]:
    """
    Crawl all faculty websites and return Documents.

    Args:
        csv_docs: List of CSV documents with faculty metadata.
        max_pages_per_site: Maximum pages to crawl per website.
        max_depth: Maximum crawl depth.
        timeout: Request timeout in seconds.
        rate_limit_delay: Delay between requests in seconds.

    Returns:
        List of LlamaIndex Documents with crawled website content.
    """
    info("[Crawling] Extracting URLs from CSV documents...")
    debug(f"Extracting URLs from {len(csv_docs)} CSV documents...")
    url_mapping = extract_urls_from_csv_docs(csv_docs)
    verbose(f"URL extraction result: {len(url_mapping)} faculty with URLs")

    if not url_mapping:
        warning("No URLs found in CSV documents")
        debug("No Website or Lab Website fields found in CSV metadata")
        return []

    info(f"Found {len(url_mapping)} faculty with websites to crawl")
    debug(f"Faculty with URLs: {list(url_mapping.keys())[:5]}...")
    verbose(f"URL mapping: {[(fid, info['faculty_name'], len(info['urls'])) for fid, info in list(url_mapping.items())[:5]]}")

    all_documents = []
    crawl_date = datetime.now().isoformat()
    debug(f"Crawl date: {crawl_date}")
    debug(f"Crawl parameters: max_pages={max_pages_per_site}, max_depth={max_depth}, timeout={timeout}s, rate_limit={rate_limit_delay}s")

    for faculty_id, faculty_info in url_mapping.items():
        faculty_name = faculty_info["faculty_name"]
        urls = faculty_info["urls"]

        info(f"Crawling websites for {faculty_name} ({faculty_id})...")
        debug(f"Faculty: {faculty_name} ({faculty_id}), URLs to crawl: {len(urls)}")
        verbose(f"URLs: {[u['url'] for u in urls]}")

        for url_info in urls:
            url = url_info["url"]
            source_type = url_info["source_type"]

            info(f"Crawling {source_type}: {url}")
            debug(f"URL: {url}, Type: {source_type}")

            try:
                pages = crawl_website(
                    url,
                    max_pages=max_pages_per_site,
                    max_depth=max_depth,
                    timeout=timeout,
                    rate_limit_delay=rate_limit_delay
                )

                info(f"Crawled {len(pages)} pages from {url}")
                debug(f"Pages: {[(p['url'], len(p['content'])) for p in pages[:3]]}")
                verbose(f"Page titles: {[p['title'] for p in pages[:5]]}")

                # Create Documents from crawled pages
                for page in pages:
                    doc = Document(
                        text=page["content"],
                        metadata={
                            "faculty_id": faculty_id,
                            "faculty_name": faculty_name,
                            "url": page["url"],
                            "page_title": page["title"],
                            "source_type": source_type,
                            "crawl_date": crawl_date,
                            "type": "faculty_website"
                        }
                    )
                    all_documents.append(doc)
                    verbose(f"Document created: {page['url']} ({len(page['content'])} chars)")

            except Exception as e:
                error(f"Error crawling {url}: {e}")
                verbose(f"Crawl exception: {type(e).__name__}: {str(e)}")
                continue

    info(f"Total: {len(all_documents)} website pages crawled from {len(url_mapping)} faculty")
    debug(f"Documents created: {len(all_documents)} pages")
    verbose(f"Documents by source type: {[doc.metadata.get('source_type') for doc in all_documents[:10]]}")
    return all_documents

