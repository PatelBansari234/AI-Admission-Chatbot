"""Turn saved website source code into readable text sections.

The raw file is a full page dump, so most of it is markup, scripts and inline
CSS. We keep only visible content and split on headings so each chunk keeps the
section it belongs to.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup

# "nav" and "footer" carry the same sitewide menu/link-dump on every page
# (confirmed on a real crawl: a faculty roster's chunk lost to retrieval
# because a several-thousand-character global footer ("Departments / Courses
# offered / Syllabi / Fee structure / ...") got appended to it, diluting the
# chunk's embedding away from the actual names it contained). The important
# links inside <nav> aren't lost - load_html() pulls those separately from
# the page's JSON-LD structured data.
_DROP_TAGS = [
    "script", "style", "noscript", "svg", "iframe", "form", "meta", "link",
    "nav", "footer",
]


def _visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    # Drop empties and leftover minified-JS/CSS lines.
    keep = [ln for ln in lines if ln and not re.fullmatch(r"[\W_]{1,4}", ln)]
    out, prev = [], None
    for ln in keep:  # collapse the repeated menu items WordPress emits
        if ln != prev:
            out.append(ln)
        prev = ln
    return "\n".join(out)


def _page_title(soup: BeautifulSoup, fallback: str) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return fallback


def load_html(path: Path) -> list[dict]:
    """Return records shaped like the PDF loader: {"text", "source", "section"}."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    title = _page_title(soup, path.stem)

    records: list[dict] = []

    # 1. Structured data (JSON-LD nav blocks) carries the real admission URLs.
    for script in soup.find_all("script", type="application/ld+json"):
        raw = (script.string or "").strip()
        if raw and ("Admission" in raw or "url" in raw):
            urls = sorted(set(re.findall(r'"url"\s*:\s*"([^"]+)"', raw)))
            names = re.findall(r'"name"\s*:\s*"([^"]+)"', raw)
            if urls:
                lines = [f"{n}: {u}" for n, u in zip(names, urls)] or urls
                records.append({
                    "text": f"Official page links for {title}:\n" + "\n".join(lines),
                    "source": path.name,
                    "section": "Site links",
                })

    body = soup.body or soup
    text = _visible_text(BeautifulSoup(str(body), "lxml"))

    # 2. Split visible text into sections using the heading text as an anchor.
    headings = [h.get_text(" ", strip=True) for h in body.find_all(["h1", "h2", "h3"])]
    headings = [h for h in headings if 3 < len(h) < 120]

    if headings:
        pattern = "|".join(re.escape(h) for h in dict.fromkeys(headings))
        parts = re.split(f"^({pattern})$", text, flags=re.MULTILINE)
        current = "Overview"
        buffer = parts[0]
        for i in range(1, len(parts), 2):
            if buffer.strip():
                records.append({"text": buffer.strip(), "source": path.name, "section": current})
            current = parts[i].strip()
            buffer = f"{current}\n{parts[i + 1]}" if i + 1 < len(parts) else current
        if buffer.strip():
            records.append({"text": buffer.strip(), "source": path.name, "section": current})
    else:
        records.append({"text": text, "source": path.name, "section": title})

    return [r for r in records if len(r["text"]) >= 60]
