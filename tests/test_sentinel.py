"""Tests for the Sentinel watchdog and its self-verifying certificates."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from faceblock import evidence, identity as ident
from faceblock import record as rec
from faceblock.faces import detect_faces, load_image
from faceblock.match import RankedMatch
from faceblock.search import Candidate
from faceblock.sentinel import IMPERSONATION, OWNED, Sighting

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "messi5.jpg"


def _match(url: str, score: float = 0.8) -> RankedMatch:
    return RankedMatch(
        candidate=Candidate(image_url=url, page_url=url),
        similarity=score,
        faces_found=1,
    )


# --- handles --------------------------------------------------------------

@pytest.mark.parametrize(
    "handle,url,owned",
    [
        ("instagram.com/megh", "https://www.instagram.com/megh/", True),
        ("instagram.com/megh", "https://instagram.com/megh/p/123", True),
        ("instagram.com/megh", "https://instagram.com/someone_else", False),
        ("instagram.com/megh", "https://x.com/megh", False),
        ("github.com", "https://github.com/anybody", True),
        ("https://x.com/megh", "https://x.com/megh/status/1", True),
    ],
)
def test_handle_ownership(handle, url, owned):
    assert ident.Handle.parse(handle).matches(url) is owned


def test_handle_rejects_unparseable():
    with pytest.raises(ValueError):
        ident.Handle.parse("")


# --- identity store -------------------------------------------------------

def test_identity_round_trip_preserves_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(ident, "IDENTITY_DIR", tmp_path)
    face = detect_faces(load_image(SAMPLE))[0]
    ident.enroll("Test Person", [face.embedding], ["instagram.com/testperson"])

    loaded = ident.load("test-person")
    assert loaded.name == "Test Person"
    assert [h.as_text() for h in loaded.handles] == ["instagram.com/testperson"]
    np.testing.assert_allclose(loaded.embeddings[0], face.embedding, atol=1e-6)
    # The reloaded reference must still recognise the same face perfectly.
    assert loaded.best_similarity(face.embedding) == pytest.approx(1.0, abs=1e-5)


def test_loading_unknown_identity_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ident, "IDENTITY_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="No enrolled identity"):
        ident.load("nobody")


# --- classification -------------------------------------------------------

def _identity() -> ident.Identity:
    return ident.Identity(
        name="Demo", slug="demo",
        handles=[ident.Handle.parse("instagram.com/real")],
    )


def test_owned_page_is_not_flagged():
    person = _identity()
    url = "https://instagram.com/real/p/1"
    s = Sighting(_match(url), OWNED if person.owns(url) else IMPERSONATION, 0.8)
    assert s.status == OWNED and s.severity == "none"


def test_unrecognised_social_page_is_high_severity():
    person = _identity()
    url = "https://instagram.com/impostor"
    assert not person.owns(url)
    assert Sighting(_match(url), IMPERSONATION, 0.8).severity == "high"


def test_unrecognised_non_social_page_is_medium_severity():
    assert Sighting(_match("https://randomblog.com/x"), IMPERSONATION, 0.8).severity == "medium"


# --- sighting records -----------------------------------------------------

def test_sighting_record_carries_sentinel_block():
    face = detect_faces(load_image(SAMPLE))[0]
    r = rec.build_sighting_record(
        SAMPLE, face, _match("https://instagram.com/impostor"),
        "Demo", IMPERSONATION, "high", None,
    )
    assert r["schema"] == "faceblock/sighting-record/v1"
    assert r["sentinel"]["identity"] == "Demo"
    assert r["sentinel"]["subject_controls_page"] is False


def test_canonical_form_contains_no_json_floats():
    """Floats would hash differently in Python and JavaScript, breaking the
    in-browser verification the certificate depends on."""
    import re

    face = detect_faces(load_image(SAMPLE))[0]
    r = rec.build_sighting_record(
        SAMPLE, face, _match("https://instagram.com/impostor"),
        "Demo", IMPERSONATION, "high", None,
    )
    assert not re.search(rb"[:,]-?\d+\.\d+", rec.canonical_json(r))


# --- certificates ---------------------------------------------------------

def _certificate(tmp_path) -> tuple[Path, str, dict]:
    face = detect_faces(load_image(SAMPLE))[0]
    r = rec.build_sighting_record(
        SAMPLE, face, _match("https://instagram.com/impostor"),
        "Demo", IMPERSONATION, "high", None,
    )
    digest = rec.compute_digest(r)
    meta = {"subject": "Demo", "url": "https://instagram.com/impostor",
            "platform": "Instagram", "severity": "high", "network": "Local (Ganache)",
            "contract": "0x" + "11" * 20, "tx_hash": "0x" + "22" * 32}
    html = evidence.build(r, digest, meta, "local")
    path = evidence.save(html, digest, tmp_path / "cert.html")
    return path, digest, r


def test_certificate_is_standalone_and_embeds_the_record(tmp_path):
    path, digest, record = _certificate(tmp_path)
    html = path.read_text()
    assert digest in html
    assert record["match"]["page_url"] in html
    # Must not depend on anything it cannot carry with it.
    assert "<script src=" not in html and "http-equiv" not in html


def test_certificate_never_embeds_raw_biometrics(tmp_path):
    path, _, record = _certificate(tmp_path)
    html = path.read_text()
    # The hash of the embedding is carried; the embedding itself must not be.
    assert record["subject"]["face_embedding_sha256"] in html
    assert not re.search(r"-?0\.\d{6,}\s*,\s*-?0\.\d{6,}", html), (
        "a float array that looks like a face embedding is present"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_canonicalisation_matches_python(tmp_path):
    """The whole certificate rests on JS and Python agreeing byte for byte."""
    _, digest, record = _certificate(tmp_path)
    record_file = tmp_path / "r.json"
    record_file.write_text(json.dumps(record, ensure_ascii=False))

    script = """
    const fs = require('fs'), crypto = require('crypto');
    const R = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
    function canonicalize(v) {
      if (v === null) return "null";
      if (Array.isArray(v)) return "[" + v.map(canonicalize).join(",") + "]";
      if (typeof v === "object") {
        return "{" + Object.keys(v).sort().map(
          k => JSON.stringify(k) + ":" + canonicalize(v[k])).join(",") + "}";
      }
      return JSON.stringify(v);
    }
    process.stdout.write(crypto.createHash('sha256')
      .update(canonicalize(R), 'utf8').digest('hex'));
    """
    script_file = tmp_path / "c.js"
    script_file.write_text(script)
    out = subprocess.run(
        ["node", str(script_file), str(record_file)],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == digest
