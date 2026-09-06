"""Parser tests against a recorded Google Lens response.

Recorded from a real SerpApi call so the parser can be exercised without
spending searches from a small monthly quota.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceblock.search import parse_lens_response

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "google_lens_response.json"


@pytest.fixture
def parsed():
    return parse_lens_response(json.loads(FIXTURE.read_text()))


def test_parses_every_entry_that_has_an_image(parsed):
    # 8 visual_matches in the fixture, one of which carries no image at all.
    assert len(parsed.candidates) == 7


def test_entry_without_an_image_is_skipped_not_fatal(parsed):
    assert all(c.image_url for c in parsed.candidates)
    assert not any("no-image" in (c.page_url or "") for c in parsed.candidates)


def test_prefers_full_image_but_keeps_thumbnail_fallback(parsed):
    instagram = next(c for c in parsed.candidates if "instagram.com/p/" in (c.page_url or ""))
    assert instagram.image_url.startswith("https://lookaside.instagram.com/")
    assert instagram.thumbnail_url.startswith("https://encrypted-tbn")


def test_recognises_social_platforms(parsed):
    found = {c.platform for c in parsed.candidates if c.platform}
    assert {"Instagram", "Facebook", "X/Twitter", "Reddit", "TikTok"} <= found


def test_non_social_pages_are_not_flagged_as_social(parsed):
    etsy = next(c for c in parsed.candidates if "etsy.com" in (c.page_url or ""))
    assert etsy.platform is None and not etsy.is_social


def test_social_candidates_are_ordered_before_others(parsed):
    social = parsed.social_candidates
    assert social and all(c.is_social for c in social)
    assert social[0].platform_kind == "social"


def test_best_guess_comes_from_related_content(parsed):
    assert parsed.best_guess == "Danny DeVito"


def test_titles_and_match_type_are_carried(parsed):
    wiki = next(c for c in parsed.candidates if "wikipedia.org" in (c.page_url or ""))
    assert wiki.page_title.startswith("Ficheiro:")
    assert wiki.match_type == "similar"


def test_api_error_is_raised_not_swallowed():
    with pytest.raises(RuntimeError, match="SerpApi error"):
        parse_lens_response({"error": "Your account has run out of searches."})


def test_empty_response_yields_no_candidates():
    result = parse_lens_response({})
    assert result.candidates == [] and result.best_guess is None
