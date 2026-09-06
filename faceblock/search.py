"""Stage 2 - genuine reverse-image search via Google Cloud Vision Web Detection.

This calls a live index over the public web and parses whatever comes back.
Nothing here is hardcoded or replayed: with no network, or an image the index
has never seen, the pipeline returns zero candidates and says so.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import (
    SEARCH_PROVIDER,
    SERPAPI_SEARCH,
    SERPAPI_UPLOAD,
    VISION_ENDPOINT,
    google_api_key,
    serpapi_key,
)

# Domains treated as social/profile surfaces. `kind` separates mainstream social
# networks from professional and developer profiles, so the pipeline can prefer a
# true "social media post" while still reporting the rest.
PLATFORMS: dict[str, tuple[str, str]] = {
    "instagram.com": ("Instagram", "social"),
    "facebook.com": ("Facebook", "social"),
    "fb.com": ("Facebook", "social"),
    "twitter.com": ("X/Twitter", "social"),
    "x.com": ("X/Twitter", "social"),
    "threads.net": ("Threads", "social"),
    "tiktok.com": ("TikTok", "social"),
    "reddit.com": ("Reddit", "social"),
    "tumblr.com": ("Tumblr", "social"),
    "pinterest.com": ("Pinterest", "social"),
    "vk.com": ("VK", "social"),
    "weibo.com": ("Weibo", "social"),
    "mastodon.social": ("Mastodon", "social"),
    "bsky.app": ("Bluesky", "social"),
    "youtube.com": ("YouTube", "social"),
    "twitch.tv": ("Twitch", "social"),
    "flickr.com": ("Flickr", "social"),
    "linkedin.com": ("LinkedIn", "profile"),
    "github.com": ("GitHub", "profile"),
    "gitlab.com": ("GitLab", "profile"),
    "medium.com": ("Medium", "profile"),
    "dev.to": ("DEV", "profile"),
    "behance.net": ("Behance", "profile"),
    "dribbble.com": ("Dribbble", "profile"),
    "about.me": ("about.me", "profile"),
    "substack.com": ("Substack", "profile"),
}


def classify_url(url: str) -> tuple[str | None, str | None]:
    """Map a URL to (platform name, kind), or (None, None) if unrecognised."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    for domain, (name, kind) in PLATFORMS.items():
        if host == domain or host.endswith("." + domain):
            return name, kind
    return None, None


@dataclass
class Candidate:
    """One page/image the reverse-image index returned for the query photo."""

    image_url: str
    page_url: str | None = None
    page_title: str | None = None
    match_type: str = "page"  # full | partial | page | similar
    thumbnail_url: str | None = None  # fallback when the full image 403s
    platform: str | None = None
    platform_kind: str | None = None

    def __post_init__(self) -> None:
        self.platform, self.platform_kind = classify_url(self.page_url or self.image_url)

    @property
    def is_social(self) -> bool:
        return self.platform_kind is not None


@dataclass
class SearchResult:
    """Everything Vision reported for one query image."""

    candidates: list[Candidate] = field(default_factory=list)
    best_guess: str | None = None
    entities: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def social_candidates(self) -> list[Candidate]:
        """Social/profile hits, mainstream social networks first."""
        hits = [c for c in self.candidates if c.is_social]
        hits.sort(key=lambda c: 0 if c.platform_kind == "social" else 1)
        return hits


def web_detection(image_path: str | Path, max_results: int = 50) -> SearchResult:
    """Run Vision Web Detection on a local image and parse the response."""
    content = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "WEB_DETECTION", "maxResults": max_results}],
            }
        ]
    }

    response = requests.post(
        VISION_ENDPOINT,
        params={"key": google_api_key()},
        json=payload,
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Vision API returned HTTP {response.status_code}: {response.text[:400]}"
        )

    body = response.json()["responses"][0]
    if "error" in body:
        raise RuntimeError(f"Vision API error: {body['error'].get('message')}")

    web = body.get("webDetection", {})
    result = SearchResult(raw=web)

    seen: set[str] = set()

    def add(candidate: Candidate) -> None:
        key = candidate.page_url or candidate.image_url
        if key not in seen:
            seen.add(key)
            result.candidates.append(candidate)

    # Pages carrying the image are the most useful hits: they give both the page
    # (which is what "a matching social media post" means) and the image on it.
    for page in web.get("pagesWithMatchingImages", []):
        images = page.get("fullMatchingImages") or page.get("partialMatchingImages") or []
        add(
            Candidate(
                image_url=images[0]["url"] if images else page["url"],
                page_url=page["url"],
                page_title=page.get("pageTitle"),
                match_type="full" if page.get("fullMatchingImages") else "partial",
            )
        )

    for image in web.get("fullMatchingImages", []):
        add(Candidate(image_url=image["url"], match_type="full"))
    for image in web.get("partialMatchingImages", []):
        add(Candidate(image_url=image["url"], match_type="partial"))
    for image in web.get("visuallySimilarImages", []):
        add(Candidate(image_url=image["url"], match_type="similar"))

    guesses = web.get("bestGuessLabels", [])
    result.best_guess = guesses[0]["label"] if guesses else None
    result.entities = [
        e["description"] for e in web.get("webEntities", []) if e.get("description")
    ]
    return result


# --- SerpApi / Google Lens backend ---------------------------------------
#
# Google Vision needs billing enabled on the Cloud project. SerpApi's free tier
# needs only a signup, so it is the default backend. It takes a public image URL
# or an uploaded image, so a local photo is POSTed to their /image endpoint
# first and searched by the id that comes back.

# SerpApi documents a 500 KB cap without saying whether that means 500,000 or
# 512,000 bytes, so aim comfortably below both.
_SERP_MAX_UPLOAD = 460_000


def _compressed_jpeg(image_path: str | Path, limit: int = _SERP_MAX_UPLOAD) -> bytes:
    """Return JPEG bytes under the upload limit, degrading quality then size.

    Face matching is done later against the ORIGINAL local file, so compressing
    the copy sent to the search provider costs nothing in match accuracy.
    """
    import cv2

    from .faces import load_image

    data = Path(image_path).read_bytes()
    if len(data) <= limit and Path(image_path).suffix.lower() in {".jpg", ".jpeg"}:
        return data

    image = load_image(image_path)
    for scale in (1.0, 0.75, 0.5, 0.35):
        resized = image
        if scale < 1.0:
            resized = cv2.resize(image, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
        for quality in (92, 80, 68, 55, 40):
            ok, buf = cv2.imencode(".jpg", resized,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok and buf.nbytes <= limit:
                return buf.tobytes()
    raise RuntimeError(
        f"Could not compress {image_path} below {limit // 1024} KB for upload"
    )


def _upload_to_serpapi(image_path: str | Path) -> str:
    """Upload a local image and return its (10-minute) image id."""
    payload = _compressed_jpeg(image_path)
    response = requests.post(
        SERPAPI_UPLOAD,
        files={"image": (Path(image_path).stem + ".jpg", payload, "image/jpeg")},
        data={"api_key": serpapi_key()},
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"SerpApi upload failed (HTTP {response.status_code}): {response.text[:300]}"
        )
    body = response.json()
    image_id = body.get("image_id")
    if not image_id:
        raise RuntimeError(f"SerpApi upload returned no image_id: {body}")
    return image_id


def google_lens(image_path: str | Path, search_type: str | None = None) -> SearchResult:
    """Reverse image search via SerpApi's Google Lens engine.

    `type` is deliberately omitted by default: the engine returns
    `visual_matches` without it, and passing an unsupported value is an easy way
    to waste one of a small monthly quota on an error.
    """
    params = {
        "engine": "google_lens",
        "image_id": _upload_to_serpapi(image_path),
        "api_key": serpapi_key(),
    }
    if search_type:
        params["type"] = search_type
    response = requests.get(SERPAPI_SEARCH, params=params, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"SerpApi search failed (HTTP {response.status_code}): {response.text[:300]}"
        )
    return parse_lens_response(response.json())


def parse_lens_response(body: dict) -> SearchResult:
    """Turn a Google Lens payload into candidates. Split out so it can be
    tested against a recorded response without spending API quota."""
    if body.get("error"):
        raise RuntimeError(f"SerpApi error: {body['error']}")

    result = SearchResult(raw=body)
    seen: set[str] = set()

    # exact_matches only appears when explicitly requested via `type`;
    # visual_matches is what the default call returns.
    for key, match_type in (("exact_matches", "full"), ("visual_matches", "similar")):
        for entry in body.get(key, []):
            page = entry.get("link")
            thumbnail = entry.get("thumbnail")
            image = entry.get("image") or thumbnail
            if not image:
                continue
            dedupe_key = page or image
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            result.candidates.append(
                Candidate(
                    image_url=image,
                    page_url=page,
                    page_title=entry.get("title"),
                    match_type=match_type,
                    thumbnail_url=thumbnail,
                )
            )

    related = body.get("related_content") or []
    result.best_guess = related[0].get("query") if related else None
    result.entities = [r["query"] for r in related if r.get("query")]
    return result


def reverse_image_search(image_path: str | Path) -> SearchResult:
    """Run whichever backend is configured."""
    if SEARCH_PROVIDER == "vision":
        return web_detection(image_path)
    if SEARCH_PROVIDER == "serpapi":
        return google_lens(image_path)
    raise RuntimeError(
        f"Unknown FACEBLOCK_SEARCH_PROVIDER {SEARCH_PROVIDER!r}; "
        "expected 'serpapi' or 'vision'"
    )
