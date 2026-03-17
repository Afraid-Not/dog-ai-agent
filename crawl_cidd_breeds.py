#!/usr/bin/env python3
"""
Crawl all breed pages linked from https://cidd.discoveryspace.ca/breeds/overview.html
and save full content to a JSON file.
"""
import json
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BREEDS_OVERVIEW_URL = "https://cidd.discoveryspace.ca/breeds/overview.html"
OUTPUT_FILE = "cidd_breeds.json"


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/") or url


def is_breed_page_url(url: str) -> bool:
    """True if URL is an individual breed page (e.g. .../breed/affenpinscher.html)."""
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    return "/breed/" in path and path.endswith(".html")


def extract_breed_links(html: str, page_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(page_url).netloc.lower()
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc.lower() != base_domain:
            continue
        absolute = normalize_url(absolute)
        if is_breed_page_url(absolute):
            links.add(absolute)
    return links


def fetch_page(url: str, session: requests.Session) -> requests.Response | None:
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"  [skip] {url}: {e}", file=sys.stderr)
        return None


def page_to_record(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = (title_tag.get_text(strip=True) or "").strip() if title_tag else ""

    for tag in soup(["script", "style"]):
        tag.decompose()
    body = soup.find("body") or soup
    text = (body.get_text(separator=" ", strip=True) or "").strip()
    text = re.sub(r"\s+", " ", text)

    return {
        "url": url,
        "title": title,
        "text": text,
        "text_length": len(text),
    }


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; CIDD-Breeds-Crawler/1.0)",
    })

    # 1) Fetch breeds overview and collect breed page URLs
    print("Fetching breeds overview...", file=sys.stderr)
    overview_resp = fetch_page(BREEDS_OVERVIEW_URL, session)
    if not overview_resp:
        print("Failed to fetch breeds overview.", file=sys.stderr)
        sys.exit(1)

    breed_urls = extract_breed_links(overview_resp.text, BREEDS_OVERVIEW_URL)
    breed_urls = sorted(breed_urls)
    print(f"Found {len(breed_urls)} breed pages.", file=sys.stderr)

    # 2) Fetch each breed page and store full content
    records = []
    for i, url in enumerate(breed_urls):
        print(f"[{i+1}/{len(breed_urls)}] {url}", file=sys.stderr)
        resp = fetch_page(url, session)
        if resp:
            records.append(page_to_record(url, resp.text))

    # 3) Build output
    output = {
        "source": BREEDS_OVERVIEW_URL,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "total_breeds": len(records),
        "breeds": records,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} breed pages to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
