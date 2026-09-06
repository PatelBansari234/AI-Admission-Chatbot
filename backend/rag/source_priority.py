"""Small, bounded reranker-score boost by source - implements "give the
official website highest priority" as a tie-breaker, not an override: a
much more relevant PDF chunk must still outrank a barely-relevant website
chunk. Capped low enough (8%) that it only matters between candidates the
reranker already scored close together - see retriever.py's retrieve(),
which sorts and applies the confidence cutoff on the *boosted* score, but
still reports the raw rerank_score to the UI for transparency.

A PDF-freshness boost (newer prospectus over older) was built and tested
alongside this, using documents.updated_at, and deliberately left out:
updated_at reflects when a file was last (re-)ingested, and a full
`ingest_all()` sweep touches every file within the same run, so it's
ingestion-order noise, not content vintage. Measured directly on this
corpus - SXC-Handbook-2025-26.pdf (the OLDER edition) scored a *higher*
freshness boost than SXCA-Handbook-2026-27.pdf (the newer one), exactly
backwards. Shipping it would have actively worsened the stale-version
problem it was meant to help with. A real fix needs an actual vintage
signal (e.g. an admin-set "supersedes" link, or parsing the year out of a
filename/document title) - not attempted here, since a wrong freshness
signal is worse than none."""

WEBSITE_BOOST = 1.08


def boost_for(metadata: dict) -> float:
    if metadata.get("doc_type") == "website":
        return WEBSITE_BOOST
    return 1.0
