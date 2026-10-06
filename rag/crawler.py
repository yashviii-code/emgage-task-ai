import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse, urldefrag

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DOMAIN = "docs.emgage.work"

START_URL = "https://docs.emgage.work/docs/"

SITEMAP_URL = "https://docs.emgage.work/sitemap.xml"

MAX_PAGES = 200

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FILE = os.path.join(BASE_DIR, "data", "crawled", "emgage_docs.json")

REQUEST_DELAY = 0.3


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/154.0 Safari/537.36"
        )
    }
)


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):

    # Remove #section from URL
    url, _ = urldefrag(url)

    # Remove trailing slash (except for the start URL)
    if url.endswith("/") and url != START_URL:
        url = url.rstrip("/")

    return url


# ============================================================
# URL VALIDATION
# ============================================================

def is_valid_doc_url(url):

    try:

        parsed = urlparse(url)

        # Must be HTTPS
        if parsed.scheme not in ["http", "https"]:
            return False

        # Must belong to Emgage docs
        if parsed.netloc != BASE_DOMAIN:
            return False

        # Only crawl documentation pages (under /docs)
        if not parsed.path.startswith("/docs"):
            return False

        # Skip tag listing pages (not useful content)
        if "/docs/tags" in parsed.path:
            return False

        # Ignore unwanted files
        ignored_extensions = (
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
            ".zip",
            ".css",
            ".js",
            ".xml",
        )

        if parsed.path.lower().endswith(
            ignored_extensions
        ):
            return False

        return True

    except Exception:

        return False


# ============================================================
# DISCOVER URLS FROM SITEMAP
# ============================================================

def get_urls_from_sitemap():

    print()
    print("=" * 70)
    print("FETCHING SITEMAP")
    print("=" * 70)

    urls = set()

    try:
        response = session.get(
            SITEMAP_URL,
            timeout=20
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Handle XML namespace
        namespace = (
            "{http://www.sitemaps.org/schemas/sitemap/0.9}"
        )

        for url_element in root.findall(
            f"{namespace}url"
        ):
            loc = url_element.find(
                f"{namespace}loc"
            )

            if loc is not None and loc.text:

                normalized = normalize_url(loc.text)

                if is_valid_doc_url(normalized):
                    urls.add(normalized)

        print(
            f"Found {len(urls)} doc URLs "
            f"from sitemap"
        )

    except Exception as e:

        print(f"Sitemap fetch failed: {e}")
        print("Falling back to link crawling only")

    return urls


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text(soup):

    # Work on a COPY so we don't destroy links
    # for extract_links
    soup_copy = BeautifulSoup(
        str(soup),
        "html.parser"
    )

    # Remove unnecessary elements
    for tag in soup_copy(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
        ]
    ):
        tag.decompose()

    # Try to identify the main documentation content
    main = (
        soup_copy.find("main")
        or soup_copy.find("article")
        or soup_copy.find("body")
    )

    if not main:
        return ""

    text = main.get_text(
        separator="\n",
        strip=True
    )

    # Remove excessive blank lines
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# EXTRACT LINKS
# ============================================================

def extract_links(soup, current_url):

    links = set()

    for tag in soup.find_all(
        "a",
        href=True
    ):

        href = tag.get("href")

        if not href:
            continue

        # Convert relative URL to absolute URL
        absolute_url = urljoin(
            current_url,
            href
        )

        # Normalize
        absolute_url = normalize_url(
            absolute_url
        )

        if is_valid_doc_url(
            absolute_url
        ):

            links.add(
                absolute_url
            )

    return links


# ============================================================
# CRAWL SINGLE PAGE
# ============================================================

def crawl_page(url):

    try:

        response = session.get(
            url,
            timeout=20
        )

        response.raise_for_status()

    except requests.RequestException as e:

        print(
            f"  ERROR: {e}"
        )

        return None, set()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Extract page title
    title_tag = soup.find("title")

    title = (
        title_tag.get_text(
            strip=True
        )
        if title_tag
        else ""
    )

    # Extract links BEFORE extracting text
    # (extract_text works on a copy now, but
    #  this is the correct order regardless)
    links = extract_links(
        soup,
        url
    )

    # Extract text
    text = extract_text(
        soup
    )

    page_data = {

        "url": url,

        "title": title,

        "text": text,

    }

    return page_data, links


# ============================================================
# MAIN CRAWLER
# ============================================================

def crawl_website():

    visited = set()

    # Start with sitemap URLs + the start URL
    sitemap_urls = get_urls_from_sitemap()

    queue = [normalize_url(START_URL)]

    # Add sitemap URLs to the queue
    for url in sitemap_urls:
        if url not in queue:
            queue.append(url)

    pages = []

    total = len(queue)

    print()
    print("=" * 70)
    print(f"CRAWLING {total} PAGES")
    print("=" * 70)

    while queue and len(visited) < MAX_PAGES:

        current_url = queue.pop(0)

        current_url = normalize_url(
            current_url
        )

        # Skip duplicate URLs
        if current_url in visited:
            continue

        visited.add(
            current_url
        )

        print(
            f"[{len(visited)}/{min(total, MAX_PAGES)}] "
            f"{current_url}"
        )

        # Crawl page
        page_data, links = crawl_page(
            current_url
        )

        if page_data:

            # Only store pages containing text
            if page_data["text"]:

                pages.append(
                    page_data
                )

                print(
                    f"  ✓ {len(page_data['text'])} chars"
                )

            else:

                print(
                    "  ⚠ No text content"
                )

        # Add discovered links
        for link in links:

            if link not in visited:

                if link not in queue:

                    queue.append(
                        link
                    )

        # Be polite to the server
        time.sleep(
            REQUEST_DELAY
        )

    return pages


# ============================================================
# SAVE DATA
# ============================================================

def save_pages(pages):

    directory = os.path.dirname(
        OUTPUT_FILE
    )

    os.makedirs(
        directory,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pages,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)

    print(
        f"Saved {len(pages)} pages"
    )

    print(
        f"File: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("EMGAGE DOCUMENTATION CRAWLER")
    print("=" * 70)

    print(
        f"Starting URL: {START_URL}"
    )

    print(
        f"Maximum pages: {MAX_PAGES}"
    )

    print()

    pages = crawl_website()

    save_pages(
        pages
    )

    print()
    print("Crawler finished.")