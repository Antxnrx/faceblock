"""Stage 4 - build the off-chain match record and its on-chain digest.

The stored file has three parts:

    record  - the match itself. This, and only this, is what the digest covers.
    digest  - sha256 over the canonical serialisation of `record`.
    anchor  - blockchain receipt, written after anchoring. Deliberately OUTSIDE
              the digest, since it cannot exist at the time the digest is taken.

Only `digest` ever reaches the chain. The record holds hashes of the photo and
the face embedding rather than the biometrics themselves, so anchoring proves
*when* a claim was made and that it has not changed since - without publishing
anyone's face, embedding, or identity to an immutable public ledger.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import FACE_MATCH_THRESHOLD, RECORD_DIR, SEARCH_PROVIDER
from .faces import Face
from .match import RankedMatch

SCHEMA = "faceblock/match-record/v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_embedding(embedding: np.ndarray) -> str:
    """Hash an embedding via a fixed byte layout so the digest is reproducible."""
    return sha256_bytes(np.asarray(embedding, dtype=np.float32).tobytes())


def canonical_json(payload: dict[str, Any]) -> bytes:
    """Deterministic serialisation - sorted keys, no insignificant whitespace.

    Any two parties must derive byte-identical input for the digest, so the
    encoding cannot depend on dict ordering or formatting choices.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


# Numeric scores are stored as fixed-precision STRINGS, never JSON floats.
# Python renders the float 1.0 as "1.0" while JavaScript renders it as "1", so a
# float anywhere in the canonical form would hash differently in the two
# languages and break independent verification. Strings remove the ambiguity.
def _fixed(value: float) -> str:
    return f"{value:.6f}"


def compute_digest(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(record))


def build_record(
    image_path: str | Path,
    face: Face,
    match: RankedMatch,
    search_best_guess: str | None = None,
) -> dict[str, Any]:
    """Assemble the match record for a verified hit."""
    image_path = Path(image_path)
    candidate = match.candidate
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subject": {
            "photo_filename": image_path.name,
            "photo_sha256": sha256_file(image_path),
            "face_bbox": list(face.bbox),
            "detector_score": _fixed(face.detector_score),
            "face_embedding_sha256": sha256_embedding(face.embedding),
        },
        "match": {
            "page_url": candidate.page_url,
            "image_url": candidate.image_url,
            # Some hosts (LinkedIn's licdn especially) sign image URLs with a
            # short expiry, so the primary URL can 404 within hours. The search
            # provider's own thumbnail is stable and keeps the claim auditable.
            "thumbnail_url": candidate.thumbnail_url,
            "page_title": candidate.page_title,
            "platform": candidate.platform,
            "platform_kind": candidate.platform_kind,
            "search_match_type": candidate.match_type,
            "faces_found_on_candidate": match.faces_found,
            "face_similarity": _fixed(match.similarity),
        },
        "method": {
            "detector": "OpenCV YuNet (face_detection_yunet_2023mar)",
            "encoder": "OpenCV SFace (face_recognition_sface_2021dec), 128-d cosine",
            "reverse_image_search": SEARCH_PROVIDER,
            "match_threshold": _fixed(FACE_MATCH_THRESHOLD),
            "search_best_guess": search_best_guess,
        },
    }


def build_sighting_record(
    image_path: str | Path,
    face: Face,
    match: RankedMatch,
    identity_name: str,
    status: str,
    severity: str,
    search_best_guess: str | None = None,
) -> dict[str, Any]:
    """A match record for a Sentinel sweep.

    Extends the base record with who was being watched for and whether the page
    is one they control - the distinction the certificate is actually asserting.
    """
    record = build_record(image_path, face, match, search_best_guess)
    record["schema"] = "faceblock/sighting-record/v1"
    record["sentinel"] = {
        "identity": identity_name,
        "status": status,
        "severity": severity,
        "subject_controls_page": status == "owned",
    }
    return record


def save(record: dict[str, Any], digest: str, path: Path | None = None) -> Path:
    """Persist a record + digest to the off-chain store."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = path or RECORD_DIR / f"{digest[:16]}.json"
    payload = {"record": record, "digest": digest, "anchor": None}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def attach_anchor(path: Path, anchor: dict[str, Any]) -> None:
    """Record the blockchain receipt alongside (but outside) the digested record."""
    payload = json.loads(Path(path).read_text())
    payload["anchor"] = anchor
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
