"""Enrolled identities - the people who have asked to be watched for.

An identity is self-enrolled: you supply your own photos and the handles you
legitimately control. Everything here stays on the local machine. Reference
embeddings are biometric data, so they never go on chain and never leave disk -
only hashes of them ever appear in an anchored record.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .config import PROJECT_ROOT

IDENTITY_DIR = Path(PROJECT_ROOT / "data" / "identities")


def _slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-")


@dataclass
class Handle:
    """A page the enrolled person legitimately controls."""

    host: str
    path_prefix: str

    @classmethod
    def parse(cls, raw: str) -> "Handle":
        """Accept 'instagram.com/megh', 'https://x.com/megh', or a bare domain."""
        candidate = raw if "//" in raw else "https://" + raw
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if not host:
            raise ValueError(f"Could not parse a host from handle {raw!r}")
        return cls(host=host, path_prefix=parsed.path.rstrip("/").lower())

    def matches(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if host != self.host and not host.endswith("." + self.host):
            return False
        # A bare domain handle claims the whole site; a path claims its subtree.
        return parsed.path.rstrip("/").lower().startswith(self.path_prefix)

    def as_text(self) -> str:
        return self.host + self.path_prefix


@dataclass
class Identity:
    """A self-enrolled person being watched for."""

    name: str
    slug: str
    handles: list[Handle] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    enrolled_at: str = ""

    def owns(self, url: str | None) -> bool:
        return bool(url) and any(h.matches(url) for h in self.handles)

    def best_similarity(self, embedding: np.ndarray) -> float:
        """Score against the closest reference photo.

        Several reference photos materially improve recall: one frontal shot
        does not represent a face across lighting, age, and pose.
        """
        if not self.embeddings:
            return -1.0
        return max(float(np.dot(embedding, ref)) for ref in self.embeddings)

    @property
    def path(self) -> Path:
        return IDENTITY_DIR / f"{self.slug}.json"


def save(identity: Identity) -> Path:
    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": identity.name,
        "slug": identity.slug,
        "enrolled_at": identity.enrolled_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "handles": [h.as_text() for h in identity.handles],
        # float32 little-endian, base64'd - compact and exactly reproducible.
        "embeddings": [
            base64.b64encode(np.asarray(e, dtype="<f4").tobytes()).decode()
            for e in identity.embeddings
        ],
    }
    identity.path.write_text(json.dumps(payload, indent=2) + "\n")
    return identity.path


def load(slug: str) -> Identity:
    path = IDENTITY_DIR / f"{slug}.json"
    if not path.exists():
        known = ", ".join(p.stem for p in IDENTITY_DIR.glob("*.json")) or "none"
        raise RuntimeError(f"No enrolled identity {slug!r}. Enrolled: {known}")
    raw = json.loads(path.read_text())
    return Identity(
        name=raw["name"],
        slug=raw["slug"],
        enrolled_at=raw.get("enrolled_at", ""),
        handles=[Handle.parse(h) for h in raw.get("handles", [])],
        embeddings=[
            np.frombuffer(base64.b64decode(e), dtype="<f4")
            for e in raw.get("embeddings", [])
        ],
    )


def enroll(name: str, embeddings: list[np.ndarray], handles: list[str]) -> Identity:
    identity = Identity(
        name=name,
        slug=_slugify(name),
        handles=[Handle.parse(h) for h in handles],
        embeddings=embeddings,
        enrolled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    save(identity)
    return identity


def list_identities() -> list[str]:
    return sorted(p.stem for p in IDENTITY_DIR.glob("*.json")) if IDENTITY_DIR.exists() else []
