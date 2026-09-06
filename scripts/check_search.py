#!/usr/bin/env python
"""Check that the configured reverse-image-search backend is usable.

    python scripts/check_search.py

For SerpApi this queries the account endpoint, which reports your remaining
quota WITHOUT spending one of your free searches. For Vision it sends a real
(free-tier) Web Detection request, since Vision has no equivalent probe.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from faceblock.config import SEARCH_PROVIDER, google_api_key, serpapi_key  # noqa: E402


def check_serpapi() -> int:
    key = serpapi_key()
    print(f"provider: serpapi (key {len(key)} chars, ending ...{key[-4:]})")
    response = requests.get(
        "https://serpapi.com/account", params={"api_key": key}, timeout=30
    )
    if response.status_code != 200:
        print(f"FAIL  HTTP {response.status_code}: {response.text[:300]}")
        return 1
    account = response.json()
    print("OK    key is valid")
    print(f"  plan:              {account.get('plan_name', '?')}")
    print(f"  searches left:     {account.get('total_searches_left', '?')}")
    print(f"  this month used:   {account.get('this_month_usage', '?')}")
    if not account.get("total_searches_left"):
        print("\nWARNING: no searches remaining on this plan.")
        return 1
    return 0


def check_vision() -> int:
    import base64

    key = google_api_key()
    print(f"provider: vision (key {len(key)} chars, ending ...{key[-4:]})")
    image = Path(__file__).resolve().parent.parent / "samples" / "messi5.jpg"
    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image.read_bytes()).decode()},
                "features": [{"type": "WEB_DETECTION", "maxResults": 5}],
            }
        ]
    }
    response = requests.post(
        "https://vision.googleapis.com/v1/images:annotate",
        params={"key": key}, json=payload, timeout=90,
    )
    if response.status_code != 200:
        body = response.json().get("error", {})
        print(f"FAIL  HTTP {response.status_code}: {body.get('message', '')[:300]}")
        if body.get("status") == "PERMISSION_DENIED":
            print("      -> the key is valid but billing is not enabled on the project")
        return 1
    web = response.json()["responses"][0].get("webDetection", {})
    print("OK    WEB_DETECTION is enabled")
    for field in ("fullMatchingImages", "pagesWithMatchingImages"):
        print(f"  {field:26}: {len(web.get(field, []))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            check_serpapi() if SEARCH_PROVIDER == "serpapi" else check_vision()
        )
    except Exception as exc:  # noqa: BLE001 - this script reports, never traces
        print(f"FAIL  {exc}")
        raise SystemExit(1)
