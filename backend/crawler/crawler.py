"""Polite, same-domain crawler for the college website.

    python -m backend.crawler.crawler --start-url https://sxca.edu.in/ --strategy astar

Three interchangeable frontier disciplines (see frontier.py) decide visit
order: BFS (level by level), DFS (follow one branch deep before backtracking),
or A* (visit the most admission-relevant pages first, via heuristics.py).
Fetched pages are saved as HTML under data/raw/websie/crawled/ and indexed
through the existing ingest pipeline, exactly like an admin-uploaded file.
On-site PDFs linked from a page (e.g. an academic calendar button) are
followed and downloaded too, into data/raw/pdfs/crawled/, and indexed the
same way as an uploaded prospectus.

Safety / etiquette built in, not optional:
- same domain only, robots.txt is honoured
- one request in flight at a time, with a delay between requests
- hard caps on page count and depth so a bad input can't turn this into a DoS
"""

import argparse
import re
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from backend.config import (
    CRAWL_DEFAULT_DELAY,
    CRAWL_DEFAULT_MAX_DEPTH,
    CRAWL_DEFAULT_MAX_PAGES,
    CRAWL_HARD_MAX_DEPTH,
    CRAWL_HARD_MAX_PAGES,
    CRAWL_TIMEOUT,
    CRAWL_USER_AGENT,
    RAW_PDF_CRAWL_DIR,
    RAW_WEB_CRAWL_DIR,
)
from backend.crawler.frontier import FRONTIERS, AStarFrontier, CrawlNode
from backend.crawler.heuristics import build_query_heuristic
from backend.utils.crawl_log import record_run

# .pdf is deliberately NOT here - on-site PDFs (an academic calendar button
# that links straight to a file instead of another page) are followed and
# indexed like any other page; see _handle_pdf below. Everything else here
# is either a dead end for text content or a real risk (executables, archives).
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".json", ".xml",
    ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar",
    ".mp4", ".mp3", ".woff", ".woff2", ".ttf",
}


def _normalize(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"


def _same_domain(url: str, root_netloc: str) -> bool:
    """True for the root domain and its subdomains (e.g. admissions.sxca.edu.in)."""
    netloc = urlparse(url).netloc.lower().removeprefix("www.")
    return netloc == root_netloc or netloc.endswith(f".{root_netloc}")


def _is_page_link(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    suffix = re.sub(r".*(\.[a-zA-Z0-9]+)$", r"\1", parsed.path).lower()
    return suffix not in SKIP_EXTENSIONS


def _extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for tag in soup.find_all("a", href=True):
        href = urljoin(base_url, tag["href"].strip())
        if _is_page_link(href):
            links.append((href, tag.get_text(" ", strip=True)[:120]))
    return links


def _slugify(url: str, root_netloc: str, seen: dict[str, int]) -> str:
    """Filename for a fetched page. Subdomains are prefixed with their host,
    since a bare path like "/" collapses to "index" for every subdomain's
    homepage - without the prefix, lms.sxca.edu.in and the main site would
    collide on "index.html" and silently overwrite one another depending on
    visit order (caught in testing: BFS clobbered the real homepage with the
    student LMS login page under the shared "index" slug)."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.strip("/") or "index"
    path_slug = re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-").lower() or "index"
    if host != root_netloc:
        host_slug = re.sub(r"[^A-Za-z0-9]+", "-", host).strip("-")
        path_slug = f"{host_slug}-{path_slug}"
    path_slug = path_slug[:150]
    seen[path_slug] = seen.get(path_slug, 0) + 1
    return path_slug if seen[path_slug] == 1 else f"{path_slug}-{seen[path_slug]}"


def _is_pdf_link(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _pdf_filename(url: str, seen: dict[str, int]) -> str:
    """PDFs already have a real filename in their URL - use it instead of
    the heading-derived page slugs, just sanitized and de-duplicated."""
    raw_name = urlparse(url).path.rsplit("/", 1)[-1] or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).strip("_") or "document.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    stem, _, ext = name.rpartition(".")
    seen[name] = seen.get(name, 0) + 1
    return name if seen[name] == 1 else f"{stem}-{seen[name]}.{ext}"


def _load_robots(start_url: str) -> urllib.robotparser.RobotFileParser:
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:
        pass  # unreachable robots.txt -> fail open, same as most browsers/crawlers
    return parser


def crawl(
    start_url: str,
    strategy: str = "astar",
    max_pages: int = CRAWL_DEFAULT_MAX_PAGES,
    max_depth: int = CRAWL_DEFAULT_MAX_DEPTH,
    delay: float = CRAWL_DEFAULT_DELAY,
    index: bool = True,
    query: str | None = None,
) -> dict:
    """Incremental, hash-aware crawler:
    - Compares sha256 hash of fetched page content against page_hashes table.
    - Unchanged pages are skipped from re-ingestion for maximum efficiency.
    - Changed pages are archived to _versions/ and incrementally re-indexed.
    - New pages are saved and indexed.
    - Preserves existing good data if any individual page fetch/index fails.
    """
    start_time = datetime.now(timezone.utc)
    if strategy not in FRONTIERS:
        raise ValueError(f"strategy must be one of {sorted(FRONTIERS)}")
    max_pages = min(max_pages, CRAWL_HARD_MAX_PAGES)
    max_depth = min(max_depth, CRAWL_HARD_MAX_DEPTH)

    root_netloc = urlparse(start_url).netloc.lower().removeprefix("www.")
    robots = _load_robots(start_url)
    if strategy == "astar" and query:
        frontier = AStarFrontier(score_fn=build_query_heuristic(query))
    else:
        frontier = FRONTIERS[strategy]()
    frontier.push(CrawlNode(url=start_url, depth=0, parent=None, anchor_text=""))

    visited: set[str] = set()
    slug_counts: dict[str, int] = {}
    pdf_name_counts: dict[str, int] = {}
    pages: list[dict] = []
    session = requests.Session()
    session.headers["User-Agent"] = CRAWL_USER_AGENT

    pages_new = 0
    pages_changed = 0
    pages_unchanged = 0
    pages_failed = 0

    # Lazy import: avoids crawler <-> rag import cycles when ingest isn't needed.
    from backend.rag.ingest import ingest_file

    while len(frontier) and len(pages) < max_pages:
        node = frontier.pop()
        key = _normalize(node.url)
        if key in visited:
            continue
        visited.add(key)

        if not _same_domain(node.url, root_netloc):
            pages.append({"url": node.url, "status": "skipped-domain", "depth": node.depth})
            continue
        if not robots.can_fetch(CRAWL_USER_AGENT, node.url):
            pages.append({"url": node.url, "status": "skipped-robots", "depth": node.depth})
            continue

        try:
            resp = session.get(node.url, timeout=CRAWL_TIMEOUT)
        except requests.RequestException as exc:
            pages_failed += 1
            pages.append({"url": node.url, "status": f"error: {exc}", "depth": node.depth})
            continue

        content_type = resp.headers.get("Content-Type", "")
        is_pdf = _is_pdf_link(node.url) or "application/pdf" in content_type

        if resp.status_code != 200 or not (is_pdf or "text/html" in content_type):
            pages.append({
                "url": node.url, "status": f"skipped-http-{resp.status_code}",
                "depth": node.depth,
            })
            continue

        content_hash = hashlib.sha256(resp.content).hexdigest()
        existing_record = repo.get_page_hash(node.url)
        is_unchanged = existing_record and existing_record.get("content_hash") == content_hash

        if is_pdf:
            doc_type = "pdf"
            name = _pdf_filename(node.url, pdf_name_counts)
            dest = RAW_PDF_CRAWL_DIR / name
            backup_dir = RAW_PDF_CRAWL_DIR / "_versions"
        else:
            doc_type = "website"
            slug = _slugify(node.url, root_netloc, slug_counts)
            dest = RAW_WEB_CRAWL_DIR / f"{slug}.html"
            backup_dir = RAW_WEB_CRAWL_DIR / "_versions"

        if is_unchanged and dest.exists():
            pages_unchanged += 1
            entry = {
                "url": node.url,
                "status": "unchanged",
                "depth": node.depth,
                "saved_as": dest.name,
                "chunks": existing_record.get("chunks", 0),
            }
            pages.append(entry)
        else:
            is_update = bool(existing_record)
            if is_update:
                pages_changed += 1
                if dest.exists():
                    backup_dir.mkdir(exist_ok=True, parents=True)
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                    shutil.copy2(dest, backup_dir / f"{dest.stem}.{stamp}{dest.suffix}")
            else:
                pages_new += 1

            if is_pdf:
                dest.write_bytes(resp.content)
            else:
                dest.write_text(resp.text, encoding="utf-8")

            entry = {
                "url": node.url,
                "status": "updated" if is_update else "new",
                "depth": node.depth,
                "saved_as": dest.name,
            }
            chunks_count = 0
            if index:
                try:
                    result = ingest_file(dest, uploaded_by=f"crawler:{strategy}", replace=True)
                    chunks_count = result.get("chunks", 0)
                    entry["chunks"] = chunks_count
                except Exception as exc:
                    pages_failed += 1
                    entry["index_error"] = str(exc)

            repo.upsert_page_hash(
                url=node.url,
                content_hash=content_hash,
                saved_as=dest.name,
                doc_type=doc_type,
                chunks=chunks_count,
            )
            pages.append(entry)

        # A PDF has no links to follow - nothing to extract from it.
        if not is_pdf and node.depth < max_depth:
            for link, anchor in _extract_links(resp.text, node.url):
                if _normalize(link) not in visited:
                    frontier.push(CrawlNode(url=link, depth=node.depth + 1, parent=node.url, anchor_text=anchor))

        if len(frontier) and len(pages) < max_pages:
            time.sleep(delay)

    end_time = datetime.now(timezone.utc)
    fetched = sum(1 for p in pages if p["status"] in ("new", "updated", "unchanged"))
    errors_count = sum(1 for p in pages if p["status"].startswith("error") or "index_error" in p)

    summary = {
        "start_url": start_url,
        "strategy": strategy,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "fetched": fetched,
        "pages_new": pages_new,
        "pages_changed": pages_changed,
        "pages_unchanged": pages_unchanged,
        "pages_failed": pages_failed,
        "skipped": sum(1 for p in pages if p["status"].startswith("skipped")),
        "errors": errors_count,
        "status": "completed" if errors_count == 0 else "completed_with_errors",
        "pages": pages,
    }
    record_run(summary)
    repo.update_crawler_settings(
        last_run_at=end_time,
        last_status="completed",
        pages_changed=pages_changed,
        pages_new=pages_new,
        pages_unchanged=pages_unchanged,
        pages_failed=pages_failed,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl the college website")
    parser.add_argument("--start-url", default="https://sxca.edu.in/")
    parser.add_argument("--strategy", choices=sorted(FRONTIERS), default="astar")
    parser.add_argument("--max-pages", type=int, default=CRAWL_DEFAULT_MAX_PAGES)
    parser.add_argument("--max-depth", type=int, default=CRAWL_DEFAULT_MAX_DEPTH)
    parser.add_argument("--delay", type=float, default=CRAWL_DEFAULT_DELAY)
    parser.add_argument("--no-index", action="store_true", help="save pages without indexing them")
    args = parser.parse_args()

    summary = crawl(
        args.start_url, args.strategy, args.max_pages, args.max_depth,
        args.delay, index=not args.no_index,
    )
    print(f"Fetched {summary['fetched']}, skipped {summary['skipped']}, errors {summary['errors']}")
    for page in summary["pages"]:
        print(f"  [{page['status']}] depth={page['depth']} {page['url']}")


if __name__ == "__main__":
    main()
