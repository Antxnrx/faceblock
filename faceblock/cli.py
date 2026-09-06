"""FaceBlock command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import chain, evidence, identity as ident, record as rec
from .config import (
    FACE_MATCH_THRESHOLD,
    MAX_CANDIDATES,
    SEARCH_PROVIDER,
    chain_preset,
)
from .faces import detect_faces, load_image, primary_face
from .match import best_verified, rank_candidates
from .search import reverse_image_search
from .sentinel import IMPERSONATION, sweep
from .term import bad, bold, dim, head, key, link, ok, warn


def _step(number: int, title: str, total: int = 5) -> None:
    print(f"\n{head(f'[{number}/{total}] {title}')}")
    print(dim("-" * (len(title) + 6)))


def cmd_run(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"{bad('error:')} no such image: {image_path}", file=sys.stderr)
        return 2

    _step(1, "Detect and encode face")
    face = primary_face(load_image(image_path))
    if face is None:
        print(f"  {bad('no face detected')} - cannot continue")
        return 1
    print(f"  bbox={key(face.bbox)}  detector_score={key(f'{face.detector_score:.3f}')}")
    print(f"  embedding: {key('128-d SFace')}, "
          f"sha256={dim(rec.sha256_embedding(face.embedding)[:32] + '...')}")

    _step(2, f"Reverse image search ({SEARCH_PROVIDER})")
    search = reverse_image_search(image_path)
    print(f"  best guess:  {key(search.best_guess or '-')}")
    print(f"  candidates:  {key(len(search.candidates))} "
          f"({key(len(search.social_candidates))} on social/profile domains)")
    if not search.candidates:
        print("\n  The index returned no matches for this photo. This is a real,")
        print("  unmodified result - the pipeline does not fabricate matches.")
        return 1
    for c in search.social_candidates[:8]:
        print(f"    [{key(c.platform)}] {dim(c.page_url or c.image_url)}")

    _step(3, "Verify candidates by re-encoding their faces")
    ranked = rank_candidates(search.candidates, face.embedding, limit=args.max_candidates)
    scored = [r for r in ranked if r.faces_found]
    print(f"  fetched {key(len(ranked))} candidates, "
          f"{key(len(scored))} contained a detectable face")
    for r in ranked[:8]:
        if r.error:
            continue
        flag = ok("MATCH") if r.verified else dim("  -  ")
        score = key(f"{r.similarity:+.4f}") if r.verified else dim(f"{r.similarity:+.4f}")
        url = r.candidate.page_url or r.candidate.image_url
        print(f"    {flag} sim={score}  {url if r.verified else dim(url)}")
    print(f"  threshold for same identity: {key(FACE_MATCH_THRESHOLD)}")

    match = best_verified(ranked)
    if match is None:
        top = ranked[0].similarity if ranked else -1
        print(f"\n  {warn('No candidate cleared the threshold')} (best was {top:+.4f}).")
        print(dim("  Reporting no match rather than anchoring an unverified claim."))
        return 1
    print(f"\n  selected: {link(match.candidate.page_url or match.candidate.image_url)}")
    print(f"  platform: {key(match.candidate.platform)}  "
          f"confidence: {bold(key(f'{match.similarity:.4f}'))}")

    _step(4, "Build off-chain record and digest")
    payload = rec.build_record(image_path, face, match, search.best_guess)
    digest = rec.compute_digest(payload)
    path = rec.save(payload, digest)
    print(f"  record: {key(path)}")
    print(f"  sha256: {key(digest)}")

    _step(5, "Anchor digest on chain")
    if args.no_anchor:
        print("  skipped (--no-anchor)")
        return 0
    receipt = chain.anchor(digest, args.network)
    rec.attach_anchor(path, receipt)
    print(f"  network:  {key(receipt['network'])} (chain id {receipt['chain_id']})")
    print(f"  contract: {key(receipt['contract'])}")
    print(f"  block:    {key(receipt['block_number'])}")
    if receipt["explorer_url"]:
        print(f"  explorer: {link(receipt['explorer_url'])}")
    print(f"\n{ok('Done.')} Verify independently with:")
    print(f"  {bold(f'python -m faceblock.cli verify {path}')}")
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    """Register a person to be watched for, with the handles they control."""
    embeddings = []
    for photo in args.photo:
        face = primary_face(load_image(photo))
        if face is None:
            print(f"  no face found in {photo} - skipping", file=sys.stderr)
            continue
        embeddings.append(face.embedding)
        print(f"  {ok('encoded')} {key(photo)}")

    if not embeddings:
        print("error: no usable reference photos", file=sys.stderr)
        return 1

    person = ident.enroll(args.name, embeddings, args.handle)
    print(f"\n{ok('Enrolled')} {bold(person.name)} as {key(person.slug)}")
    print(f"  reference photos: {key(len(person.embeddings))}")
    print(f"  handles owned:    "
          f"{key(', '.join(h.as_text() for h in person.handles) or 'none')}")
    print(f"  stored at:        {dim(person.path)}")
    print(dim("\nReference embeddings stay on this machine and never go on chain."))
    return 0


def cmd_sentinel(args: argparse.Namespace) -> int:
    """Sweep the web for an enrolled face and notarise unrecognised sightings."""
    person = ident.load(args.identity)
    image_path = Path(args.photo)
    if not image_path.exists():
        print(f"{bad('error:')} no such image: {image_path}", file=sys.stderr)
        return 2

    print(f"{head('Sweeping for ' + person.name)} "
          f"({key(len(person.handles))} owned handles)")
    result = sweep(person, image_path, limit=args.max_candidates)

    print(f"\n  candidates from reverse image search: {key(len(result.search.candidates))}")
    print(f"  confirmed as this face:               {key(len(result.sightings))}")
    print(f"    on pages they control:              {ok(len(result.owned))}")
    print(f"    on pages they do NOT control:       "
          f"{bad(len(result.impersonations)) if result.impersonations else ok(0)}")

    for sighting in result.owned:
        print(f"\n  [{ok('OK')}] {dim(sighting.url)}")
        print(dim(f"       owned handle, similarity {sighting.identity_similarity:.4f}"))

    if not result.impersonations:
        print(f"\n{ok('No unrecognised sightings.')} Nothing to notarise.")
        return 0

    face = primary_face(load_image(image_path))
    targets = result.impersonations[: args.top] if args.top else result.impersonations
    if args.top and len(result.impersonations) > args.top:
        print(f"\n  notarising the top {args.top} of {len(result.impersonations)} "
              f"(raise with --top, 0 for all)")

    written: list[Path] = []
    for sighting in targets:
        tag = (bad if sighting.severity == "high" else warn)(sighting.severity.upper())
        print(f"\n  [{tag}] {link(sighting.url)}")
        print(f"       platform: {key(sighting.ranked.candidate.platform or 'unrecognised')}"
              f"  similarity: {bold(key(f'{sighting.identity_similarity:.4f}'))}")

        payload = rec.build_sighting_record(
            image_path, face, sighting.ranked, person.name,
            IMPERSONATION, sighting.severity, result.search.best_guess,
        )
        digest = rec.compute_digest(payload)
        record_path = rec.save(payload, digest)
        print(f"       record: {key(record_path.name)}  digest: {dim(digest[:24] + '...')}")

        meta = {
            "subject": person.name, "url": sighting.url,
            "platform": sighting.ranked.candidate.platform,
            "severity": sighting.severity, "network": chain_preset(args.network).name,
            "contract": "", "tx_hash": "",
        }
        if not args.no_anchor:
            receipt = chain.anchor(digest, args.network)
            rec.attach_anchor(record_path, receipt)
            meta |= {"contract": receipt["contract"], "tx_hash": receipt["tx_hash"]}
            print(f"       {ok('anchored')} in block {key(receipt['block_number'])}")
            if receipt["explorer_url"]:
                print(f"       {link(receipt['explorer_url'])}")

        cert = evidence.save(
            evidence.build(payload, digest, meta, args.network), digest
        )
        written.append(cert)
        print(f"       certificate: {key(cert)}")

    print(f"\n{ok(bold(f'{len(written)} evidence certificate(s) written.'))} "
          f"Open one in a browser -")
    print(dim("it re-checks its own digest and the chain with no server involved."))
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    """Regenerate a certificate from an existing record."""
    payload = rec.load(args.record)
    record, digest, anchor = payload["record"], payload["digest"], payload.get("anchor")
    sentinel = record.get("sentinel", {})
    meta = {
        "subject": sentinel.get("identity", "unknown"),
        "url": record["match"]["page_url"] or record["match"]["image_url"],
        "platform": record["match"]["platform"],
        "severity": sentinel.get("severity", "medium"),
        "network": (anchor or {}).get("network", chain_preset(args.network).name),
        "contract": (anchor or {}).get("contract", ""),
        "tx_hash": (anchor or {}).get("tx_hash", ""),
    }
    path = evidence.save(evidence.build(record, digest, meta, args.network), digest)
    print(f"certificate written: {path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    payload = rec.load(args.record)
    stored = payload["digest"]
    recomputed = rec.compute_digest(payload["record"])

    print(head("Local integrity"))
    print(f"  stored digest:     {key(stored)}")
    print(f"  recomputed digest: "
          f"{key(recomputed) if stored == recomputed else bad(recomputed)}")
    if stored != recomputed:
        print(f"  {bad('FAIL')} - the record has been modified since it was digested.")
        print(f"\n{bad(bold('TAMPERED: this record no longer matches its digest.'))}")
        return 1
    print(f"  {ok('OK')} - record matches its digest.")

    print(f"\n{head('On-chain anchor')}")
    found = chain.lookup(stored, args.network)
    preset = chain_preset(args.network)
    if found is None:
        print(f"  {bad('FAIL')} - digest is not anchored on {preset.name}.")
        return 1
    print(f"  network:   {key(preset.name)}")
    print(f"  contract:  {key(found['contract'])}")
    print(f"  submitter: {key(found['submitter'])}")
    print(f"  anchored:  block timestamp {key(found['timestamp'])}")
    print(f"  {ok('OK')} - digest is anchored on chain and matches the local record.")
    print(f"\n{ok(bold('VERIFIED: this record is byte-identical to the one anchored on chain.'))}")
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    address = chain.deploy(args.network)
    print("\nAdd this to your .env:")
    print(f"  FACEBLOCK_CONTRACT_ADDRESS={address}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that credentials, funds, and models are in place before a live demo."""
    from .config import contract_address, google_api_key, serpapi_key

    key_probe = serpapi_key if SEARCH_PROVIDER == "serpapi" else google_api_key
    healthy = True
    for label, probe in (
        (f"Search key ({SEARCH_PROVIDER})", lambda: f"set ({len(key_probe())} chars)"),
        ("Testnet account", lambda: chain.balance_report(args.network)),
        ("Contract address", contract_address),
    ):
        try:
            print(f"  {ok('OK')}   {label}: {key(probe())}")
        except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
            healthy = False
            print(f"  {bad('FAIL')} {label}: {exc}")
    return 0 if healthy else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="faceblock",
        description="Face encoding -> reverse image search -> blockchain anchor.",
    )
    parser.add_argument(
        "--network", default=None, help="amoy (default), sepolia, or local"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline on a photo")
    run.add_argument("image", help="path to the input photo")
    run.add_argument("--no-anchor", action="store_true", help="skip the on-chain write")
    run.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="re-hash a record and check it on chain")
    verify.add_argument("record", help="path to a record JSON file")
    verify.set_defaults(func=cmd_verify)

    deploy = sub.add_parser("deploy", help="deploy the HashAnchor contract")
    deploy.set_defaults(func=cmd_deploy)

    doctor = sub.add_parser("doctor", help="check credentials, funds, and config")
    doctor.set_defaults(func=cmd_doctor)

    enroll = sub.add_parser("enroll", help="register a face to be watched for")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--photo", action="append", required=True,
                        help="reference photo (repeatable - more is better)")
    enroll.add_argument("--handle", action="append", default=[],
                        help="a page this person controls, e.g. instagram.com/them")
    enroll.set_defaults(func=cmd_enroll)

    sentinel = sub.add_parser(
        "sentinel", help="sweep the web for an enrolled face and notarise fakes")
    sentinel.add_argument("--identity", required=True, help="enrolled slug")
    sentinel.add_argument("--photo", required=True, help="photo to search with")
    sentinel.add_argument("--no-anchor", action="store_true")
    sentinel.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    sentinel.add_argument(
        "--top", type=int, default=5,
        help="only notarise the N most severe sightings (0 = all). Each one "
             "costs a transaction, so the default keeps a demo run brisk.")
    sentinel.set_defaults(func=cmd_sentinel)

    ev = sub.add_parser("evidence", help="regenerate a certificate from a record")
    ev.add_argument("record")
    ev.set_defaults(func=cmd_evidence)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI surfaces errors, not tracebacks
        print(f"\n{bad('error:')} {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
