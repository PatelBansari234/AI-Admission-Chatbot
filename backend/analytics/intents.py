"""Group logged questions into intent clusters by embedding similarity.

Short admission questions are entity-heavy, so raw cosine similarity doesn't
cleanly separate "same intent, different phrasing" from "same entity,
different intent": in testing, "How many seats are there for BA Economics?"
vs "What is the duration of the BA Economics course?" scored *higher*
similarity (0.81) than a genuine paraphrase pair like "What scholarships are
available?" vs "Tell me about scholarship options" (0.84 - barely above).
There is no single threshold that perfectly separates these two cases with
this embedding model.

We therefore bias toward precision over recall: INTENT_DISTANCE_THRESHOLD is
set conservatively (backend/config.py) so a cluster only forms when
confidence is high, accepting that some true paraphrases end up as separate,
similar-looking clusters rather than risk silently merging two different
questions into one (which would hide a real gap from the admin).
"""

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from backend.config import INTENT_DISTANCE_THRESHOLD
from backend.db import repo
from backend.embeddings.embedder import embed_documents

_cache: dict = {"key": None, "result": None}


def compute_intents(
    distance_threshold: float = INTENT_DISTANCE_THRESHOLD, force: bool = False
) -> list[dict]:
    """Return intent clusters sorted by frequency, most-asked first.

    Cached against the chat log's row count (the log is append-only, so a
    stable count means nothing new to re-embed) so repeated admin-panel loads
    don't re-embed and re-cluster every logged question each time.
    """
    key = repo.count_chat()
    if not force and key == _cache["key"]:
        return _cache["result"]

    rows = repo.list_chat()
    if not rows:
        _cache.update(key=key, result=[])
        return []

    questions = [r["question"] for r in rows]
    embeddings = np.array(embed_documents(questions))

    if len(questions) == 1:
        labels = [0]
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(embeddings)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    result = []
    for idxs in clusters.values():
        cluster_rows = [rows[i] for i in idxs]
        cluster_emb = embeddings[idxs]
        centroid = cluster_emb.mean(axis=0)
        # The row closest to the centroid reads better as a label than an
        # arbitrary member or a synthetic average-of-strings.
        medoid_pos = int(np.argmax(cluster_emb @ centroid))
        representative = cluster_rows[medoid_pos]["question"]

        phrasing_counts = Counter(r["question"] for r in cluster_rows)
        grounded = sum(1 for r in cluster_rows if r.get("grounded"))
        timestamps = sorted(r["created_at"] for r in cluster_rows if r.get("created_at"))

        result.append({
            "representative": representative,
            "count": len(cluster_rows),
            "distinct_phrasings": len(phrasing_counts),
            "samples": [{"question": q, "count": n} for q, n in phrasing_counts.most_common(8)],
            "grounded_rate": round(grounded / len(cluster_rows), 2),
            "first_asked": timestamps[0] if timestamps else None,
            "last_asked": timestamps[-1] if timestamps else None,
        })

    result.sort(key=lambda c: c["count"], reverse=True)
    _cache.update(key=key, result=result)
    return result
