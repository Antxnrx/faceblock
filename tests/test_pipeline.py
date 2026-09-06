"""Offline tests for the parts of the pipeline that must not silently drift.

Network-dependent stages (Vision, the chain) are exercised separately by
`faceblock.cli doctor` and by an actual run; what matters here is that hashing
is deterministic, tampering is detectable, and match selection honours the
threshold rather than always producing a "match".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faceblock import record as rec
from faceblock.config import FACE_MATCH_THRESHOLD
from faceblock.faces import Face, detect_faces, load_image, similarity
from faceblock.match import RankedMatch, best_verified
from faceblock.search import Candidate, classify_url

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "messi5.jpg"


# --- stage 1: detection and encoding -------------------------------------

def test_detects_face_and_produces_unit_embedding():
    faces = detect_faces(load_image(SAMPLE))
    assert faces, "expected at least one face in the sample image"
    face = faces[0]
    assert face.embedding.shape == (128,)
    assert face.embedding.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(face.embedding), 1.0, atol=1e-5)


def test_identical_faces_match_and_unrelated_embeddings_do_not():
    face = detect_faces(load_image(SAMPLE))[0]
    assert similarity(face.embedding, face.embedding) == pytest.approx(1.0, abs=1e-5)

    # A random unit vector stands in for an unrelated identity; it must fall
    # well below the same-identity threshold.
    rng = np.random.default_rng(0)
    other = rng.normal(size=128).astype(np.float32)
    other /= np.linalg.norm(other)
    assert similarity(face.embedding, other) < FACE_MATCH_THRESHOLD


# --- stage 2: URL classification -----------------------------------------

@pytest.mark.parametrize(
    "url,platform,kind",
    [
        ("https://www.instagram.com/p/xyz/", "Instagram", "social"),
        ("https://x.com/user/status/1", "X/Twitter", "social"),
        ("https://in.linkedin.com/in/someone", "LinkedIn", "profile"),
        ("https://example.org/photo.jpg", None, None),
        ("https://notinstagram.com/p/xyz", None, None),
    ],
)
def test_classify_url(url, platform, kind):
    assert classify_url(url) == (platform, kind)


# --- stage 3: match selection --------------------------------------------

def _ranked(similarity_score: float, page: str) -> RankedMatch:
    return RankedMatch(
        candidate=Candidate(image_url=page, page_url=page),
        similarity=similarity_score,
        faces_found=1,
    )


def test_below_threshold_yields_no_match():
    below = FACE_MATCH_THRESHOLD - 0.05
    assert best_verified([_ranked(below, "https://instagram.com/p/a")]) is None


def test_social_page_preferred_over_bare_image_at_similar_confidence():
    social = _ranked(0.70, "https://instagram.com/p/a")
    other = _ranked(0.72, "https://example.com/b.jpg")
    assert best_verified([social, other]).candidate.page_url == "https://instagram.com/p/a"


def test_no_faces_on_candidate_is_never_a_match():
    empty = RankedMatch(candidate=Candidate(image_url="https://example.com/x.jpg"))
    assert best_verified([empty]) is None


# --- stage 4: digest determinism and tamper detection --------------------

def test_digest_is_key_order_independent():
    a = {"b": 2, "a": {"y": 1, "x": [3, 4]}}
    b = {"a": {"x": [3, 4], "y": 1}, "b": 2}
    assert rec.compute_digest(a) == rec.compute_digest(b)


def test_record_round_trips_and_tampering_is_detected(tmp_path):
    face = detect_faces(load_image(SAMPLE))[0]
    match = _ranked(0.81, "https://instagram.com/p/demo")
    payload = rec.build_record(SAMPLE, face, match, "demo")
    digest = rec.compute_digest(payload)

    path = rec.save(payload, digest, tmp_path / "record.json")
    stored = rec.load(path)
    assert stored["digest"] == digest
    assert rec.compute_digest(stored["record"]) == digest

    # Attaching the chain receipt must NOT change the digest: the anchor is
    # deliberately outside the digested region.
    rec.attach_anchor(path, {"tx_hash": "0xdeadbeef"})
    reloaded = rec.load(path)
    assert reloaded["anchor"]["tx_hash"] == "0xdeadbeef"
    assert rec.compute_digest(reloaded["record"]) == digest

    # Changing the matched URL must invalidate the digest.
    reloaded["record"]["match"]["page_url"] = "https://instagram.com/p/someone-else"
    path.write_text(json.dumps(reloaded, indent=2))
    tampered = rec.load(path)
    assert rec.compute_digest(tampered["record"]) != tampered["digest"]


def test_record_contains_no_raw_biometrics():
    """The anchored record must carry hashes, never the embedding itself."""
    face = detect_faces(load_image(SAMPLE))[0]
    payload = rec.build_record(SAMPLE, face, _ranked(0.8, "https://x.com/p/1"), None)
    blob = json.dumps(payload)
    assert "face_embedding_sha256" in blob
    # No float from the embedding should appear anywhere in the serialised record.
    assert str(round(float(face.embedding[0]), 6)) not in blob


def test_detection_is_safe_from_many_threads():
    """Regression: OpenCV's detector/recogniser are not thread-safe, and the
    candidate ranker calls them from a thread pool. Unserialised access
    segfaulted the process rather than raising."""
    from concurrent.futures import ThreadPoolExecutor

    image = load_image(SAMPLE)
    with ThreadPoolExecutor(max_workers=16) as pool:
        counts = list(pool.map(lambda _: len(detect_faces(image)), range(120)))
    assert set(counts) == {1}, f"inconsistent results across threads: {set(counts)}"
