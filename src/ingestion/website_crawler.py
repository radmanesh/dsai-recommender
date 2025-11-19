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


def _is_internal_link(url: str, base_domain: str) -> bool:
    """Check if a URL is internal to the base domain."""
    try:
        parsed = urlparse(url)
        link_domain = parsed.netloc.lower()
        base_domain_lower = base_domain.lower()

        # Remove www. prefix for comparison
        if link_domain.startswith("www."):
            link_domain = link_domain[4:]
        if base_domain_lower.startswith("www."):
            base_domain_lower = base_domain_lower[4:]

        return link_domain == base_domain_lower or link_domain == "" or link_domain.endswith("." + base_domain_lower)
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
    except Exception as e:
        print(f"  ⚠ Invalid URL {base_url}: {e}")
        return []

    visited: Set[str] = set()
    to_visit: List[tuple[str, int]] = [(base_url, 0)]  # (url, depth)
    pages: List[Dict[str, str]] = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    while to_visit and len(pages) < max_pages:
        current_url, depth = to_visit.pop(0)

        # Skip if already visited or too deep
        if current_url in visited or depth > max_depth:
            continue

        visited.add(current_url)

        try:
            # Rate limiting
            if len(pages) > 0:
                time.sleep(rate_limit_delay)

            # Fetch page
            response = requests.get(current_url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                continue

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ""

            # Extract text content
            text_content = _extract_text_from_html(str(soup))

            # Skip if content is too short (likely not useful)
            if len(text_content) < 100:
                continue

            pages.append({
                "url": current_url,
                "title": title,
                "content": text_content
            })

            # Extract links for further crawling
            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = _normalize_url(href, current_url)

                    # Only follow internal links
                    if _is_internal_link(absolute_url, base_domain) and absolute_url not in visited:
                        # Avoid common non-content URLs
                        if not any(skip in absolute_url.lower() for skip in [
                            '#', 'mailto:', 'tel:', 'javascript:', '.pdf', '.doc', '.docx',
                            '.zip', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js'
                        ]):
                            to_visit.append((absolute_url, depth + 1))

        except requests.exceptions.RequestException as e:
            print(f"    ⚠ Error fetching {current_url}: {e}")
            continue
        except Exception as e:
            print(f"    ⚠ Error processing {current_url}: {e}")
            continue

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
    print(f"\n[Crawling] Extracting URLs from CSV documents...")
    url_mapping = extract_urls_from_csv_docs(csv_docs)

    if not url_mapping:
        print("  ⚠ No URLs found in CSV documents")
        return []

    print(f"  ✓ Found {len(url_mapping)} faculty with websites to crawl")

    all_documents = []
    crawl_date = datetime.now().isoformat()

    for faculty_id, faculty_info in url_mapping.items():
        faculty_name = faculty_info["faculty_name"]
        urls = faculty_info["urls"]

        print(f"\n  Crawling websites for {faculty_name} ({faculty_id})...")

        for url_info in urls:
            url = url_info["url"]
            source_type = url_info["source_type"]

            print(f"    Crawling {source_type}: {url}")

            try:
                pages = crawl_website(
                    url,
                    max_pages=max_pages_per_site,
                    max_depth=max_depth,
                    timeout=timeout,
                    rate_limit_delay=rate_limit_delay
                )

                print(f"      ✓ Crawled {len(pages)} pages")

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

            except Exception as e:
                print(f"      ✗ Error crawling {url}: {e}")
                continue

    print(f"\n  ✓ Total: {len(all_documents)} website pages crawled")
    return all_documents

