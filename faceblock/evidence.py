"""Self-verifying evidence certificates.

The output of a sweep is only useful if someone else - a platform's trust and
safety team, a lawyer, a police officer - can confirm it without trusting the
person who handed it to them. So a certificate is a single standalone HTML file
that carries the record inside it and checks itself when opened:

  1. re-derives the canonical byte form of the record, in the browser
  2. recomputes its SHA-256 with WebCrypto
  3. compares that to the digest the certificate claims
  4. calls the anchoring contract directly over JSON-RPC

No server, no dependencies, no trust in this pipeline. Change one character of
the embedded record and the page fails its own audit on the next open.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eth_utils import function_signature_to_4byte_selector as _selector

from .config import PROJECT_ROOT, chain_preset

EVIDENCE_DIR = PROJECT_ROOT / "data" / "evidence"

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FaceBlock Evidence Certificate</title>
<style>
:root {
  --bg: #f6f7f9; --card: #ffffff; --ink: #14181f; --muted: #5c6674;
  --line: #e2e6ec; --ok: #0f7b46; --bad: #c02626; --wait: #8a6d1f;
  --okbg: #e8f6ee; --badbg: #fdeaea; --waitbg: #fdf6e3;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1216; --card: #171b21; --ink: #e8ecf1; --muted: #9aa5b3;
    --line: #262c35; --okbg: #10281a; --badbg: #2a1414; --waitbg: #29230f;
    --ok: #4ade80; --bad: #f87171; --wait: #fbbf24;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1216; --card: #171b21; --ink: #e8ecf1; --muted: #9aa5b3;
  --line: #262c35; --okbg: #10281a; --badbg: #2a1414; --waitbg: #29230f;
  --ok: #4ade80; --bad: #f87171; --wait: #fbbf24;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  padding: 32px 20px;
}
.wrap { max-width: 860px; margin: 0 auto; }
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 12px; padding: 24px; margin-bottom: 18px;
}
h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.07em;
     color: var(--muted); margin: 0 0 14px; font-weight: 600; }
.sub { color: var(--muted); font-size: 13px; margin: 0; }
#banner {
  border-radius: 12px; padding: 20px 24px; margin-bottom: 18px;
  border: 1px solid var(--line); background: var(--waitbg);
}
#banner .state { font-size: 21px; font-weight: 700; color: var(--wait); }
#banner.ok  { background: var(--okbg); }  #banner.ok  .state { color: var(--ok); }
#banner.bad { background: var(--badbg); } #banner.bad .state { color: var(--bad); }
#banner .why { font-size: 13px; color: var(--muted); margin-top: 4px; }
table { width: 100%; border-collapse: collapse; }
td { padding: 9px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
td:first-child { color: var(--muted); width: 190px; padding-right: 16px; }
tr:last-child td { border-bottom: 0; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 12.5px; word-break: break-all; }
ol.checks { list-style: none; padding: 0; margin: 0; }
ol.checks li {
  padding: 11px 0; border-bottom: 1px solid var(--line);
  display: flex; gap: 12px; align-items: flex-start;
}
ol.checks li:last-child { border-bottom: 0; }
.mark { width: 20px; flex: none; font-weight: 700; text-align: center; }
.mark.ok { color: var(--ok); } .mark.bad { color: var(--bad); }
.mark.wait { color: var(--wait); }
.detail { font-size: 12.5px; color: var(--muted); margin-top: 3px; }
.sev { display: inline-block; padding: 2px 9px; border-radius: 99px;
       font-size: 11.5px; font-weight: 600; }
.sev.high { background: var(--badbg); color: var(--bad); }
.sev.medium { background: var(--waitbg); color: var(--wait); }
pre { background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
      padding: 14px; overflow-x: auto; font-size: 12px; margin: 0; }
a { color: inherit; }
.foot { color: var(--muted); font-size: 12.5px; }
.foot strong { color: var(--ink); }
</style>
</head>
<body>
<div class="wrap">

  <div class="card" style="margin-bottom:18px">
    <h1>FaceBlock Evidence Certificate</h1>
    <p class="sub">A notarised record that an enrolled face appeared on a page the
    subject does not control. This document verifies itself &mdash; see below.</p>
  </div>

  <div id="banner">
    <div class="state" id="state">Verifying&hellip;</div>
    <div class="why" id="why">Recomputing the digest and querying the chain.</div>
  </div>

  <div class="card">
    <h2>Sighting</h2>
    <table>
      <tr><td>Subject</td><td id="f-subject"></td></tr>
      <tr><td>Found at</td><td class="mono"><a id="f-url" target="_blank" rel="noopener"></a></td></tr>
      <tr><td>Platform</td><td id="f-platform"></td></tr>
      <tr><td>Assessment</td><td id="f-sev"></td></tr>
      <tr><td>Face similarity</td><td id="f-sim" class="mono"></td></tr>
      <tr><td>Recorded</td><td id="f-when" class="mono"></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Independent verification</h2>
    <ol class="checks" id="checks">
      <li><span class="mark wait" data-m="1">&middot;</span><div>
        <div>Re-derive the canonical byte form of the record</div>
        <div class="detail" data-d="1">waiting</div></div></li>
      <li><span class="mark wait" data-m="2">&middot;</span><div>
        <div>Recompute SHA-256 in this browser</div>
        <div class="detail mono" data-d="2">waiting</div></div></li>
      <li><span class="mark wait" data-m="3">&middot;</span><div>
        <div>Compare against the digest this certificate claims</div>
        <div class="detail mono" data-d="3">waiting</div></div></li>
      <li><span class="mark wait" data-m="4">&middot;</span><div>
        <div>Confirm the digest is anchored on __NETWORK__</div>
        <div class="detail" data-d="4">waiting</div></div></li>
    </ol>
  </div>

  <div class="card">
    <h2>Anchor</h2>
    <table>
      <tr><td>Network</td><td>__NETWORK__ (chain id __CHAINID__)</td></tr>
      <tr><td>Contract</td><td class="mono">__CONTRACT__</td></tr>
      <tr><td>Digest</td><td class="mono">__DIGEST__</td></tr>
      <tr><td>Transaction</td><td class="mono" id="f-tx"></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>The record, verbatim</h2>
    <pre id="f-record"></pre>
  </div>

  <div class="card foot">
    <p style="margin-top:0"><strong>What this proves.</strong> That this exact record
    existed in this exact form at the block time shown, and has not been altered
    since. The digest is derived from the record itself, so any edit &mdash; a
    changed URL, a nudged score &mdash; breaks the match on the next open.</p>
    <p><strong>What it does not prove.</strong> That the match is correct. Face
    recognition is probabilistic and degrades on pose, lighting, age, and
    lookalikes. This certifies what the pipeline concluded and when, not that the
    conclusion is true. Treat it as evidence to review, never as a verdict.</p>
    <p style="margin-bottom:0"><strong>Privacy.</strong> No photograph or face
    embedding is stored on chain or in this file &mdash; only their hashes. The
    ledger holds a single 32-byte digest and nothing else.</p>
  </div>
</div>

<script>
const RECORD    = __RECORD__;
const DIGEST    = "__DIGEST__";
const RPC       = "__RPC__";
const CONTRACT  = "__CONTRACT__";
const SEL_IS    = "__SEL_IS__";
const SEL_GET   = "__SEL_GET__";
const META      = __META__;

// Must match faceblock/record.py canonical_json(): keys sorted, no whitespace.
// Every numeric field in the record is a fixed-precision STRING precisely so
// that Python and JavaScript cannot disagree about how to render a float.
function canonicalize(v) {
  if (v === null) return "null";
  if (Array.isArray(v)) return "[" + v.map(canonicalize).join(",") + "]";
  if (typeof v === "object") {
    return "{" + Object.keys(v).sort().map(
      k => JSON.stringify(k) + ":" + canonicalize(v[k])).join(",") + "}";
  }
  return JSON.stringify(v);
}

function mark(n, state, text) {
  const m = document.querySelector(`[data-m="${n}"]`);
  const d = document.querySelector(`[data-d="${n}"]`);
  m.className = "mark " + (state === "ok" ? "ok" : state === "bad" ? "bad" : "wait");
  m.textContent = state === "ok" ? "\\u2713" : state === "bad" ? "\\u2717" : "\\u00b7";
  if (text) d.textContent = text;
}

function verdict(ok, state, why) {
  const b = document.getElementById("banner");
  b.className = ok ? "ok" : "bad";
  document.getElementById("state").textContent = state;
  document.getElementById("why").textContent = why;
}

async function rpc(data) {
  const res = await fetch(RPC, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "eth_call",
      params: [{ to: CONTRACT, data: data }, "latest"] })
  });
  return await res.json();
}

(async function () {
  // Fill the human-readable half first, so the page is useful even if the
  // chain is unreachable from wherever it is opened.
  document.getElementById("f-subject").textContent  = META.subject;
  document.getElementById("f-platform").textContent = META.platform || "\\u2014";
  document.getElementById("f-sim").textContent      = RECORD.match.face_similarity;
  document.getElementById("f-when").textContent     = RECORD.created_at;
  document.getElementById("f-tx").textContent       = META.tx_hash || "not anchored";
  const sev = document.getElementById("f-sev");
  sev.innerHTML = `<span class="sev ${META.severity}">${META.severity.toUpperCase()}</span>`;
  const a = document.getElementById("f-url");
  a.textContent = META.url; a.href = META.url;
  document.getElementById("f-record").textContent = JSON.stringify(RECORD, null, 2);

  const canon = canonicalize(RECORD);
  mark(1, "ok", canon.length + " bytes");

  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canon));
  const got = [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
  mark(2, "ok", got);

  if (got !== DIGEST) {
    mark(3, "bad", "does not match " + DIGEST);
    mark(4, "bad", "skipped - the record is already invalid");
    verdict(false, "TAMPERED",
      "The record does not hash to the digest this certificate claims. It has been altered.");
    return;
  }
  mark(3, "ok", "identical");

  try {
    const isRes = await rpc(SEL_IS + DIGEST);
    if (isRes.error) throw new Error(isRes.error.message);
    if (BigInt(isRes.result) === 0n) {
      mark(4, "bad", "this digest is not anchored on " + META.network);
      verdict(false, "NOT ANCHORED",
        "The record is internally consistent, but no anchor for it exists on chain.");
      return;
    }
    const g = await rpc(SEL_GET + DIGEST);
    const hex = (g.result || "").replace(/^0x/, "");
    const ts = Number(BigInt("0x" + (hex.slice(0, 64) || "0")));
    const submitter = "0x" + hex.slice(88, 128);
    mark(4, "ok", "anchored " + new Date(ts * 1000).toUTCString() + " by " + submitter);
    verdict(true, "VERIFIED",
      "This record hashes to a digest anchored on " + META.network + ". It has not been altered since.");
  } catch (e) {
    // A browser blocked by CORS is not evidence of a bad record - say so plainly
    // rather than implying the certificate failed.
    mark(4, "wait", "could not reach the chain from this browser (" + e.message + ")");
    verdict(false, "PARTIALLY VERIFIED",
      "The record matches its digest. The on-chain check could not run here - verify it manually with the contract and digest above.");
    document.getElementById("banner").className = "";
  }
})();
</script>
</body>
</html>
"""


def build(
    record: dict[str, Any],
    digest: str,
    meta: dict[str, Any],
    network: str | None = None,
) -> str:
    """Render a standalone certificate for one anchored sighting."""
    preset = chain_preset(network)
    return (
        _TEMPLATE.replace("__RECORD__", json.dumps(record, ensure_ascii=False))
        .replace("__META__", json.dumps(meta, ensure_ascii=False))
        .replace("__DIGEST__", digest)
        .replace("__NETWORK__", preset.name)
        .replace("__CHAINID__", str(preset.chain_id))
        .replace("__CONTRACT__", meta.get("contract", "not deployed"))
        .replace("__RPC__", preset.rpc_urls[0])
        .replace("__SEL_IS__", "0x" + _selector("isAnchored(bytes32)").hex())
        .replace("__SEL_GET__", "0x" + _selector("get(bytes32)").hex())
    )


def save(html: str, digest: str, path: Path | None = None) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = path or EVIDENCE_DIR / f"evidence-{digest[:16]}.html"
    path.write_text(html, encoding="utf-8")
    return path
