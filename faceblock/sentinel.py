"""The watchdog: find every place an enrolled face appears, and split the
pages that person controls from the ones they do not.

A face match on a page the subject owns is unremarkable. A face match on a page
they have never heard of is the thing worth notarising - a profile using their
photo without their knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .identity import Identity
from .match import RankedMatch, rank_candidates
from .search import SearchResult, reverse_image_search

OWNED = "owned"
IMPERSONATION = "impersonation"


@dataclass
class Sighting:
    """One confirmed appearance of the enrolled face somewhere on the web."""

    ranked: RankedMatch
    status: str
    identity_similarity: float

    @property
    def url(self) -> str:
        return self.ranked.candidate.page_url or self.ranked.candidate.image_url

    @property
    def severity(self) -> str:
        """How much a sighting should worry the subject.

        An unrecognised face on a social platform is a plausible fake profile.
        The same face on an aggregator or scraper site is usually just a mirror
        of a legitimate post - still worth logging, far less alarming.
        """
        if self.status == OWNED:
            return "none"
        return "high" if self.ranked.candidate.platform_kind == "social" else "medium"


@dataclass
class SweepResult:
    identity: Identity
    search: SearchResult
    sightings: list[Sighting] = field(default_factory=list)

    @property
    def owned(self) -> list[Sighting]:
        return [s for s in self.sightings if s.status == OWNED]

    @property
    def impersonations(self) -> list[Sighting]:
        """Unrecognised sightings, most alarming first."""
        hits = [s for s in self.sightings if s.status == IMPERSONATION]
        hits.sort(key=lambda s: (s.severity != "high", -s.identity_similarity))
        return hits


def sweep(identity: Identity, image_path: str | Path, limit: int | None = None) -> SweepResult:
    """Search the web for the enrolled face and classify what comes back."""
    search = reverse_image_search(image_path)
    reference = identity.embeddings[0] if identity.embeddings else None
    if reference is None:
        raise RuntimeError(f"Identity {identity.slug!r} has no reference embedding")

    kwargs = {"limit": limit} if limit else {}
    ranked = rank_candidates(search.candidates, reference, **kwargs)

    result = SweepResult(identity=identity, search=search)
    for entry in ranked:
        if not entry.verified:
            continue  # not this person's face; nothing to classify
        result.sightings.append(
            Sighting(
                ranked=entry,
                status=OWNED if identity.owns(entry.candidate.page_url) else IMPERSONATION,
                identity_similarity=entry.similarity,
            )
        )
    return result
