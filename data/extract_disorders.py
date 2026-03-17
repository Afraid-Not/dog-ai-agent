#!/usr/bin/env python3
"""
1. Fetch disorder overview page and collect all official disorder names (link text).
2. For each breed, split text by section:
   - 유전병 (genetic_disorders): sections 1–3 (Most Important, Other disorders with increased incidence, Disorders associated with conformation)
   - 질병 (other_disorders): section 4 (Other disorders which may be inherited)
3. Save breed -> genetic_disorders, other_disorders to cidd_breed_disorders.json.
"""
import json
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

DISORDER_OVERVIEW_URL = "https://cidd.discoveryspace.ca/disorder/overview.html"
BREEDS_JSON = "cidd_breeds.json"
OUTPUT_JSON = "cidd_breed_disorders.json"


def get_disorder_names(session: requests.Session) -> list[str]:
    """Fetch disorder overview HTML and return all disorder names (link text for /disorder/*.html)."""
    resp = session.get(DISORDER_OVERVIEW_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    base_domain = urlparse(DISORDER_OVERVIEW_URL).netloc
    names = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or not href.endswith(".html"):
            continue
        absolute = urljoin(DISORDER_OVERVIEW_URL, href)
        if "/disorder/" not in absolute or urlparse(absolute).netloc != base_domain:
            continue
        text = (a.get_text(strip=True) or "").strip()
        # Skip nav/category labels that are not disorder names
        if not text or text in names:
            continue
        if text in ("Disorders", "Breeds", "Home"):
            continue
        if text.startswith("Inherited ") and text.endswith("disorders"):
            continue
        names.append(text)
    # Sort by length descending so we match longer names first (e.g. "Diabetes mellitus" before "Mellitus")
    names.sort(key=len, reverse=True)
    return names


# Section headers in breed page (in order). First 3 = 유전병, 4th = 질병(기타 가능성 유전질환).
SECTION_MARKERS = [
    "Most Important",
    "Other disorders which have an increased incidence in this breed",
    "Disorders associated with conformation",
    "Other disorders which may be inherited in this breed",
]
# Text after the last section (we stop here for section 4)
FOOTER_MARKER = "For more information about this breed"


def split_breed_text_by_sections(text: str) -> tuple[str, str]:
    """
    Split breed page text into:
    - genetic_text: sections 1–3 (유전병: inherited / breed predisposition)
    - other_text: section 4 (질병: may be inherited, reported sporadically)
    """
    genetic_parts = []
    other_text = ""
    lower = text.lower()
    # Find positions of each marker (case-insensitive)
    positions = []
    for m in SECTION_MARKERS:
        idx = lower.find(m.lower())
        positions.append((idx, m))
    footer_idx = lower.find(FOOTER_MARKER.lower())

    for i, (idx, _) in enumerate(positions):
        if idx < 0:
            continue
        # End of this section = start of next marker or footer
        if i + 1 < len(positions) and positions[i + 1][0] >= 0:
            end = positions[i + 1][0]
        else:
            end = footer_idx if footer_idx >= 0 else len(text)
        block = text[idx:end].strip() if end >= 0 else text[idx:].strip()
        if i < 3:
            genetic_parts.append(block)
        else:
            other_text = block

    genetic_text = " ".join(genetic_parts)
    return genetic_text, other_text


def find_disorders_in_text(text: str, disorder_names: list[str]) -> list[str]:
    """Return list of disorder names that appear in text. Prefer longer (more specific) names."""
    found = []
    text_lower = text.lower()
    for name in disorder_names:
        if name.lower() in text_lower:
            name_lower = name.lower()
            if not any(
                name_lower != f.lower() and name_lower in f.lower()
                for f in found
            ):
                found.append(name)
    return found


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CIDD-Extract/1.0)"})

    print("Fetching disorder list from overview...", file=sys.stderr)
    disorder_names = get_disorder_names(session)
    print(f"Found {len(disorder_names)} disorder names.", file=sys.stderr)

    with open(BREEDS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    breed_name_from_title = re.compile(r"^(.+?)\s*\|\s*University", re.I)

    results = []
    for i, breed in enumerate(data["breeds"]):
        title = breed.get("title", "")
        text = breed.get("text", "")
        m = breed_name_from_title.match(title)
        name = m.group(1).strip() if m else title
        genetic_text, other_text = split_breed_text_by_sections(text)
        genetic_disorders = find_disorders_in_text(genetic_text, disorder_names)
        other_disorders = find_disorders_in_text(other_text, disorder_names)
        results.append({
            "breed": name,
            "url": breed.get("url", ""),
            "genetic_disorders": sorted(genetic_disorders),   # 유전병 (sections 1–3)
            "other_disorders": sorted(other_disorders),       # 질병 (section 4: may be inherited)
        })
        print(
            f"[{i+1}/{len(data['breeds'])}] {name}: 유전병 {len(genetic_disorders)}, 질병 {len(other_disorders)}",
            file=sys.stderr,
        )

    output = {
        "source": BREEDS_JSON,
        "disorder_list_source": DISORDER_OVERVIEW_URL,
        "note": "genetic_disorders=유전병(1~3섹션), other_disorders=질병(4섹션, 산발 보고)",
        "total_breeds": len(results),
        "breeds": results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUTPUT_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
