"""Runtime configuration, loaded from the environment (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_DIR = Path(
    os.getenv("FACEBLOCK_MODEL_DIR", str(Path.home() / ".faceblock" / "models"))
)
RECORD_DIR = Path(
    os.getenv("FACEBLOCK_RECORD_DIR", str(PROJECT_ROOT / "data" / "records"))
)

# Cosine threshold above which two faces are treated as the same person.
#
# SFace's published same-identity threshold is 0.363, and that is too permissive
# here. Measured on a real run: a genuine match scored 0.94 while an unrelated
# look-alike scored 0.3639 - clearing the published threshold by 0.0009. The
# candidate pool for any one photo is drawn from visually similar images, so
# near-threshold scores are exactly where the false positives live.
#
# 0.45 sits in the empty band between those two populations: every genuine match
# observed scored 0.94 or above, and the highest false positive scored 0.3639.
FACE_MATCH_THRESHOLD = float(os.getenv("FACEBLOCK_FACE_THRESHOLD", "0.45"))

# Cap on candidate pages we download and re-rank, to stay polite and bounded.
MAX_CANDIDATES = int(os.getenv("FACEBLOCK_MAX_CANDIDATES", "25"))

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
SERPAPI_SEARCH = "https://serpapi.com/search"
SERPAPI_UPLOAD = "https://serpapi.com/image"

# Which reverse-image-search backend to use. SerpApi (Google Lens) needs only a
# free-tier key with no card; Google Vision needs billing enabled on the project.
SEARCH_PROVIDER = os.getenv("FACEBLOCK_SEARCH_PROVIDER", "serpapi").lower()


@dataclass(frozen=True)
class ChainPreset:
    """A preconfigured EVM testnet.

    `rpc_urls` is an ordered fallback list: public testnet RPCs go down or
    rate-limit without warning, and a demo should not hinge on one endpoint.
    """

    name: str
    chain_id: int
    rpc_urls: tuple[str, ...]
    explorer: str
    faucet: str
    currency: str


CHAIN_PRESETS: dict[str, ChainPreset] = {
    "amoy": ChainPreset(
        name="Polygon Amoy",
        chain_id=80002,
        rpc_urls=(
            "https://polygon-amoy-bor-rpc.publicnode.com",
            "https://rpc-amoy.polygon.technology",
        ),
        explorer="https://amoy.polygonscan.com",
        faucet="https://faucet.polygon.technology/",
        currency="POL",
    ),
    # A throwaway in-memory chain for proving the pipeline end to end without a
    # faucet. Real transactions and receipts, but no public explorer - use a
    # testnet for anything a judge or third party needs to verify.
    "local": ChainPreset(
        name="Local (Ganache)",
        chain_id=1337,
        rpc_urls=("http://127.0.0.1:8545",),
        explorer="",
        faucet="ganache pre-funds accounts; no faucet needed",
        currency="ETH",
    ),
    "sepolia": ChainPreset(
        name="Ethereum Sepolia",
        chain_id=11155111,
        rpc_urls=(
            "https://ethereum-sepolia-rpc.publicnode.com",
            "https://rpc.sepolia.org",
        ),
        explorer="https://sepolia.etherscan.io",
        faucet="https://sepoliafaucet.com/",
        currency="ETH",
    ),
}

DEFAULT_NETWORK = os.getenv("FACEBLOCK_NETWORK", "amoy")


def chain_preset(network: str | None = None) -> ChainPreset:
    """Resolve a network name to its preset, honouring an RPC URL override."""
    key = (network or DEFAULT_NETWORK).lower()
    if key not in CHAIN_PRESETS:
        raise ValueError(
            f"Unknown network {key!r}. Known networks: {', '.join(CHAIN_PRESETS)}"
        )
    preset = CHAIN_PRESETS[key]
    override = os.getenv("FACEBLOCK_RPC_URL")
    if override:
        # Public RPCs rate-limit; let users point at their own node/provider.
        preset = ChainPreset(**{**preset.__dict__, "rpc_urls": (override,)})
    return preset


def google_api_key() -> str:
    key = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_VISION_API_KEY is not set. Copy .env.example to .env and add "
            "a Cloud Vision API key (see README, 'Configure credentials')."
        )
    return key


def serpapi_key() -> str:
    key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SERPAPI_API_KEY is not set. Get a free key (no card required) at "
            "https://serpapi.com/users/sign_up and put it in .env"
        )
    return key


def private_key() -> str:
    key = os.getenv("FACEBLOCK_PRIVATE_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FACEBLOCK_PRIVATE_KEY is not set. Use a throwaway TESTNET key only - "
            "never a key holding real funds."
        )
    return key if key.startswith("0x") else "0x" + key


def contract_address() -> str:
    addr = os.getenv("FACEBLOCK_CONTRACT_ADDRESS", "").strip()
    if not addr:
        raise RuntimeError(
            "FACEBLOCK_CONTRACT_ADDRESS is not set. Deploy the contract first: "
            "`python -m faceblock.cli deploy`"
        )
    return addr
