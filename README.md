# FaceBlock

A pipeline that encodes a face from a photo, finds where that face genuinely
appears on the public web via reverse-image search, and anchors the result on a
public blockchain as a tamper-evident record.

```
photo ──▶ [1] detect + encode face      YuNet detector, SFace 128-d embedding
      ──▶ [2] reverse image search      SerpApi / Google Lens (live index; Vision optional)
      ──▶ [3] verify candidates         re-encode each hit, cosine-match the face
      ──▶ [4] build record + digest     canonical JSON, SHA-256
      ──▶ [5] anchor digest on chain    HashAnchor contract on Polygon Amoy
```

Nothing in the search path is stubbed or replayed. If the index has never seen
the photo, the run reports **no match** and exits non-zero rather than inventing
one.

## Sentinel - the part that makes this worth building

The obvious reading of the brief builds a tool that finds a stranger. We built
the inverse: a tool that proves **someone is pretending to be you**.

You enrol your own face and the handles you actually control. A sweep then finds
every place your face appears online and splits the results in two - pages you
own, and pages you do not. The second list is the interesting one: fake
profiles, catfish accounts, scam listings using your photo. Each is anchored on
chain as timestamped, tamper-evident evidence.

```bash
python -m faceblock.cli enroll --name "Your Name" \
    --photo me.jpg --photo me2.jpg \
    --handle instagram.com/you --handle linkedin.com/in/you

python -m faceblock.cli sentinel --identity your-name --photo me.jpg
```

```
  confirmed as this face:               4
    on pages they control:              3
    on pages they do NOT control:       1

  [HIGH] https://instagram.com/not_actually_you
         platform: Instagram  similarity: 0.7412
         anchored in block 8912431
         certificate: data/evidence/evidence-ca46e87e6ce54210.html
```

This inversion matters for three reasons:

1. **The chain becomes load-bearing.** Anchoring a search result nobody disputes
   is decoration. Anchoring *proof that a fake profile existed on a given date*
   is the artefact a platform takedown or a police report actually needs, and it
   has to be tamper-evident to be worth anything.
2. **The consent problem disappears.** You are searching for your own face,
   against handles you declared. There is no non-consenting subject anywhere in
   the flow.
3. **Wrong matches get cheaper, not more dangerous.** A false positive here
   produces a page for *you* to review, not an accusation published about a
   stranger.

### Self-verifying evidence certificates

Evidence is worthless if the recipient has to trust whoever handed it over. So
each sighting produces a single standalone HTML file that **audits itself when
opened**:

1. re-derives the canonical byte form of the record, in the browser
2. recomputes its SHA-256 with WebCrypto
3. compares that against the digest the certificate claims
4. calls the anchoring contract directly over JSON-RPC

No server, no dependencies, no trust in this pipeline. A trust-and-safety
reviewer opens one file and sees `VERIFIED`. Change a single character of the
embedded record and the page reports `TAMPERED` on the next open, showing the
digest that no longer matches.

Making that work required one non-obvious decision: **every numeric field in a
record is stored as a fixed-precision string, never a JSON float.** Python
renders the float `1.0` as `1.0` and JavaScript renders it as `1`, so a single
float anywhere in the canonical form would hash differently in the two languages
and the in-browser check would fail. A test asserts the canonical form contains
no floats, and another re-hashes a record under Node to confirm the two
languages agree byte for byte.

## Which blockchain

**Polygon Amoy testnet** (chain id `80002`) by default; `--network sepolia`
switches to Ethereum Sepolia. Both are free — testnet tokens come from a faucet,
and every transaction is independently checkable on a public block explorer.

The contract is [`contracts/HashAnchor.sol`](contracts/HashAnchor.sol) — 53 lines,
one storage mapping:

```solidity
function anchor(bytes32 digest) external;                    // write once, never overwrite
function get(bytes32 digest) external view returns (uint64, address);
function isAnchored(bytes32 digest) external view returns (bool);
```

### Why only a hash goes on chain

This is the central design decision, and it is deliberate.

A public ledger is immutable and world-readable. Writing "this face belongs to
this person's Instagram account" directly onto it would publish an **irrevocable
claim of identity about a real person** — one that cannot be corrected if the
match is wrong, and cannot be deleted if the subject never consented.

So the match record stays **off-chain** in `data/records/`, and only the SHA-256
digest of that record is anchored. Anyone holding the record can recompute the
digest and check it against the chain. That yields exactly the property the
brief asks for — a tamper-evident, independently verifiable record — without
permanently publishing anyone's biometrics or identity.

The record itself stores the **hash** of the photo and the **hash** of the face
embedding, never the raw biometrics.

## Setup

Requires Python 3.10+.

```bash
git clone <your-repo-url> && cd faceblock
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The two ONNX models (YuNet ~0.2 MB, SFace ~39 MB) download automatically on
first run into `~/.faceblock/models/`.

### Configure credentials

```bash
cp .env.example .env
```

**`GOOGLE_VISION_API_KEY`** — create a Google Cloud project, enable the **Cloud
Vision API**, and create an API key. Web Detection's first 1,000 units/month are
free, but the project still needs **billing enabled** for the API to activate.

**`FACEBLOCK_PRIVATE_KEY`** — a **throwaway** testnet account. Never use a key
holding real funds; `.env` is plaintext and this key only needs faucet tokens.
Fund it at <https://faucet.polygon.technology/>.

Then deploy the contract and record its address in `.env`:

```bash
python -m faceblock.cli deploy
# -> FACEBLOCK_CONTRACT_ADDRESS=0x...
```

Confirm everything is wired up:

```bash
python -m faceblock.cli doctor
```

### Testing without a funded account

Public testnets need faucet tokens, which can take time to obtain. To exercise
the full chain path immediately - real deploys, real transactions, real receipts
- run against a local chain instead:

```bash
npx ganache --wallet.deterministic --chain.chainId 1337 --port 8545   # terminal 1

export FACEBLOCK_PRIVATE_KEY=0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d
python -m faceblock.cli --network local deploy
export FACEBLOCK_CONTRACT_ADDRESS=<deployed address>
python -m faceblock.cli --network local run photo.jpg
```

That key is ganache's well-known deterministic test account - it is public by
design and holds nothing. A local chain has no public explorer, so use Amoy for
anything a third party needs to verify independently.

### Sample photos

Only `samples/messi5.jpg` (from the OpenCV sample data) ships with this repo.
Photos used during development are deliberately not redistributed: one is a
personal photograph, and the rest are copyrighted press images. Bring your own
and point the CLI at it.

## Running it

```bash
python -m faceblock.cli run path/to/photo.jpg
```

Each stage prints what it found. Add `--no-anchor` to run the detection and
search stages without touching the chain.

### Verifying a record

This is the payoff — the record can be checked by anyone, without trusting the
machine that produced it:

```bash
python -m faceblock.cli verify data/records/<digest>.json
```

It recomputes the digest from the record and compares it to what is anchored on
chain. **Edit any field in the record and re-run it** — the digest changes, the
comparison fails, and the tampering is caught. That demonstration is worth
including in the screen recording.

## How the confidence score is computed

Vision reports where a *visually similar image* appears. It does not establish
that the face on that page is the same person — so the pipeline does not take
its word for it.

Stage 3 fetches each candidate image, runs the **same** detector and encoder
used on the query photo, and computes cosine similarity between the embeddings.
Only candidates at or above **0.363** — SFace's published same-identity
threshold — are eligible. The score written into the record is computed here, by
this pipeline, not supplied by the search provider.

If no candidate clears the threshold, the run reports no match and anchors
nothing. Anchoring an unverified claim would defeat the point of the record.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

30 offline tests cover embedding shape and normalisation, URL classification,
threshold behaviour (including that a below-threshold candidate yields *no*
match), digest determinism, tamper detection, handle-ownership rules, identity
round-tripping, certificate self-containment, the guarantee that no raw
biometric data reaches a record or a certificate, and a cross-language check
that Node and Python hash the same record identically.

## Known limitations

**Precision depends on how well-indexed the subject is**

This is the single most important limitation, and it is measured rather than
assumed. Two real runs:

| Subject | Result |
| --- | --- |
| Heavily-indexed public figure | 8 matches at 0.94-0.98, no false positives, correct identification |
| Private individual, one indexed photo | correct match at 0.94, but a look-alike cleared the 0.363 threshold by 0.0009 |

With thousands of crawled copies the true matches dominate and scores separate
cleanly. With a single crawled photo the candidate pool fills with look-alikes
and the threshold has to do work it is not really good enough to do. Precision
scales with indexed footprint - it is not uniform across people.

**Face matching**
- SFace is a lightweight CPU model. It degrades on extreme pose, low light,
  heavy occlusion, and large age gaps, and — like all face recognition — is
  weakest exactly where a false match is most damaging: identical twins, close
  relatives, and strong lookalikes.
- Published accuracy is not uniform across demographic groups. A single cosine
  threshold does not correct for that.
- The 0.363 threshold trades precision against recall at a fixed point. It is
  tunable via `FACEBLOCK_FACE_THRESHOLD`, and no single value is right for
  every deployment.

**Reverse image search**
- Vision only searches what Google has crawled. Private accounts, follower-only
  posts, and recently uploaded images are invisible to it. A "no match" result
  means *not found in this index* — never *not on the internet*.
- Coverage skews toward images that have been widely republished, which biases
  results toward public figures.
- SerpApi's free tier is capped (about 100 searches/month) and its Google Lens
  results are not identical to Vision's - the two backends will disagree on
  edge cases. Whichever ran is recorded in the anchored record.

**Blockchain**
- Anchoring proves a record existed in a given form at a given block time and
  has not changed since. It says **nothing** about whether the match is correct.
  A confidently wrong match, anchored, is still wrong — permanently.
- Testnet only. Amoy and Sepolia carry no value guarantee and can be reset by
  their operators; production use would need a mainnet or a durable timestamping
  service.
- Anchors are write-once by design: re-running on an unchanged record is
  rejected on chain rather than silently overwritten.

**Sentinel specifically**
- A sweep only finds impersonations that reuse *your actual photo*. An impostor
  who uses a different picture of you, or an AI-generated face, will not be
  found - see the reverse-image-search limits above.
- "Pages you do not control" is not the same as "impersonation". Legitimate
  reposts, press coverage, and aggregator mirrors all land in that bucket and
  need human review. Severity is a triage hint, not a verdict.
- Handle ownership is self-declared and unverified. Anyone can claim any handle
  at enrolment; the tool trusts the operator about their own accounts.

**Scope and consent**
- This demo is run against a **consenting subject**. It is a demonstration of a
  technical mechanism, not a deployable identity-verification product.
- Running this pipeline against people who have not consented raises real legal
  exposure — biometric-privacy statutes such as Illinois BIPA and the EU GDPR's
  Article 9 treatment of biometric data both apply to exactly this kind of
  processing. Any real deployment needs a consent framework, a retention and
  deletion policy, an appeals path for wrong matches, and legal review first.
- There is no rate limiting, authentication, or audit trail around who runs a
  search. Those are prerequisites for any multi-user deployment.

## Layout

| Path | Purpose |
| --- | --- |
| [`faceblock/faces.py`](faceblock/faces.py) | Stage 1 — detection, encoding, similarity |
| [`faceblock/search.py`](faceblock/search.py) | Stage 2 — SerpApi + Vision backends, platform classification |
| [`faceblock/match.py`](faceblock/match.py) | Stage 3 — candidate fetch, re-encode, rank |
| [`faceblock/record.py`](faceblock/record.py) | Stage 4 — canonical JSON, SHA-256 digest |
| [`faceblock/chain.py`](faceblock/chain.py) | Stage 5 — deploy, anchor, look up |
| [`faceblock/identity.py`](faceblock/identity.py) | Enrolled people, owned handles, reference embeddings |
| [`faceblock/sentinel.py`](faceblock/sentinel.py) | Sweep and owned-vs-impersonation classification |
| [`faceblock/evidence.py`](faceblock/evidence.py) | Self-verifying HTML certificates |
| [`faceblock/cli.py`](faceblock/cli.py) | Orchestration (`run`, `sentinel`, `enroll`, `evidence`, `verify`, `deploy`, `doctor`) |
| [`contracts/HashAnchor.sol`](contracts/HashAnchor.sol) | The anchoring contract |
| [`contracts/HashAnchor.json`](contracts/HashAnchor.json) | Committed ABI + bytecode, so no solc is needed to deploy |
