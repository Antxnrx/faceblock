"""Stage 3 - verify candidates by re-encoding their faces.

Vision tells us where a visually similar image appears; it does not tell us that
the face on that page is the same person. So we fetch each candidate image, run
the same detector/encoder used on the query photo, and score the pair with
cosine similarity. That score - computed by this pipeline, not by the search
provider - is the confidence written into the match record.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import requests

from .config import FACE_MATCH_THRESHOLD, MAX_CANDIDATES
from .faces import detect_faces, load_image
from .search import Candidate

# Some CDNs reject requests without a browser-ish UA.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FaceBlock/0.1; +verification-pipeline)"}
_MAX_BYTES = 12 * 1024 * 1024


@dataclass
class RankedMatch:
    """A candidate scored against the query face."""

    candidate: Candidate
    similarity: float = -1.0
    faces_found: int = 0
    error: str | None = None

    @property
    def verified(self) -> bool:
        """True when the candidate's best face matches the query face."""
        return self.similarity >= FACE_MATCH_THRESHOLD


def _fetch(url: str) -> bytes:
    response = requests.get(url, headers=_HEADERS, timeout=30, stream=True)
    response.raise_for_status()
    chunks, total = [], 0
    for chunk in response.iter_content(chunk_size=1 << 16):
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_BYTES:
            raise ValueError(f"image exceeds {_MAX_BYTES // (1024 * 1024)} MB cap")
    return b"".join(chunks)


def _score(candidate: Candidate, query: np.ndarray) -> RankedMatch:
    result = RankedMatch(candidate=candidate)
    # Many full-size URLs sit behind hotlink protection (lookaside.instagram.com
    # and friends), while the search provider's own thumbnail is always
    # fetchable. A smaller crop still carries enough face for a usable score.
    sources = [candidate.image_url]
    if candidate.thumbnail_url and candidate.thumbnail_url != candidate.image_url:
        sources.append(candidate.thumbnail_url)

    faces, errors = [], []
    for url in sources:
        try:
            faces = detect_faces(load_image(_fetch(url)))
            errors = []
            break
        except Exception as exc:  # noqa: BLE001 - try the fallback, then report
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        result.error = "; ".join(errors)
        return result

    result.faces_found = len(faces)
    if faces:
        # A page may show several people; the match is the best-scoring face.
        result.similarity = max(float(np.dot(query, f.embedding)) for f in faces)
    return result


def rank_candidates(
    candidates: list[Candidate], query: np.ndarray, limit: int = MAX_CANDIDATES
) -> list[RankedMatch]:
    """Score candidates against the query embedding, best first."""
    subset = candidates[:limit]
    if not subset:
        return []

    with ThreadPoolExecutor(max_workers=8) as pool:
        ranked = list(pool.map(lambda c: _score(c, query), subset))

    ranked.sort(key=lambda r: r.similarity, reverse=True)
    return ranked


def best_verified(ranked: list[RankedMatch]) -> RankedMatch | None:
    """Highest-scoring verified match, preferring a real social/profile page."""
    verified = [r for r in ranked if r.verified]
    if not verified:
        return None
    # Already sorted by similarity; a social hit outranks a bare image URL at
    # comparable confidence, since the deliverable asks for a social media post.
    verified.sort(key=lambda r: (0 if r.candidate.is_social else 1, -r.similarity))
    return verified[0]
