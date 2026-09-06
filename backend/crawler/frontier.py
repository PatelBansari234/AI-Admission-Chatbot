"""Frontier data structures backing the three crawl strategies.

Same crawl loop, three container disciplines:
- BFS:  FIFO queue  -> explores the site level by level.
- DFS:  LIFO stack  -> follows one branch (e.g. Admissions -> UG -> ...) deep
        before backtracking to siblings.
- A*:   priority heap ordered by heuristic relevance minus a depth penalty
        -> visits the most admission-relevant pages first regardless of level.
"""

import heapq
import itertools
from collections import deque
from dataclasses import dataclass

from backend.crawler.heuristics import score_page

DEPTH_PENALTY = 1.5  # per level, so A* still prefers shallower pages when tied


@dataclass
class CrawlNode:
    url: str
    depth: int
    parent: str | None
    anchor_text: str


class BFSFrontier:
    def __init__(self):
        self._q = deque()

    def push(self, node: CrawlNode) -> None:
        self._q.append(node)

    def pop(self) -> CrawlNode:
        return self._q.popleft()

    def __len__(self) -> int:
        return len(self._q)


class DFSFrontier:
    def __init__(self):
        self._stack: list[CrawlNode] = []

    def push(self, node: CrawlNode) -> None:
        self._stack.append(node)

    def pop(self) -> CrawlNode:
        return self._stack.pop()

    def __len__(self) -> int:
        return len(self._stack)


class AStarFrontier:
    """Best-first search: highest score() pops first.

    score_fn defaults to the fixed admission-keyword heuristic, but the
    on-demand live-crawl fallback (backend/crawler/live_search.py) passes in
    heuristics.build_query_heuristic(question) instead, so the crawl chases
    whatever a specific unanswered question is actually about.
    """

    def __init__(self, score_fn=score_page):
        self._heap: list[tuple[float, int, CrawlNode]] = []
        self._counter = itertools.count()  # tie-breaker so heapq never compares CrawlNode
        self._score_fn = score_fn

    def push(self, node: CrawlNode) -> None:
        priority = -(self._score_fn(node.url, node.anchor_text) - DEPTH_PENALTY * node.depth)
        heapq.heappush(self._heap, (priority, next(self._counter), node))

    def pop(self) -> CrawlNode:
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)


FRONTIERS = {"bfs": BFSFrontier, "dfs": DFSFrontier, "astar": AStarFrontier}
