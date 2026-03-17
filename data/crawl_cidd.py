#!/usr/bin/env python3
"""
Crawl https://cidd.discoveryspace.ca/ and its same-domain subpages,
then save the collected data to a JSON file.
"""
import json
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://cidd.discoveryspace.ca/"
OUTPUT_FILE = "cidd_subpages.json"


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def is_same_domain(url: str, base_domain: str) -> bool:
    return get_domain(url) == base_domain


def normalize_url(url: str) -> str:
    """Remove fragment and normalize."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/") or url


def extract_internal_links(html: str, page_url: str) -> set[str]:
    base_domain = get_domain(page_url)
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absolute = urljoin(page_url, href)
        if not is_same_domain(absolute, base_domain):
            continue
        links.add(normalize_url(absolute))
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

    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.decompose()
    body = soup.find("body") or soup
    text = (body.get_text(separator=" ", strip=True) or "").strip()
    text = re.sub(r"\s+", " ", text)[:50000]  # limit size

    return {
        "url": url,
        "title": title,
        "text_preview": text[:2000] if text else "",
        "text_length": len(text),
    }


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; CIDD-Crawler/1.0)",
    })

    # 1) Fetch main page and collect internal links
    print("Fetching main page...", file=sys.stderr)
    main_resp = fetch_page(BASE_URL, session)
    if not main_resp:
        print("Failed to fetch main page.", file=sys.stderr)
        sys.exit(1)

    all_urls = {normalize_url(BASE_URL)}
    all_urls |= extract_internal_links(main_resp.text, BASE_URL)

    # 2) Fetch each subpage and build records
    records = []
    urls_list = sorted(all_urls)
    for i, url in enumerate(urls_list):
        print(f"[{i+1}/{len(urls_list)}] {url}", file=sys.stderr)
        if url == normalize_url(BASE_URL):
            resp = main_resp
        else:
            resp = fetch_page(url, session)
        if resp:
            records.append(page_to_record(url, resp.text))

    # 3) Build output
    output = {
        "source": BASE_URL,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "total_pages": len(records),
        "subpages": records,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} pages to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
