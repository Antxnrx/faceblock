"""Stage 5 - anchor the digest on an EVM testnet, and verify it later.

Only a 32-byte digest is sent. Nothing in a transaction here reveals the photo,
the embedding, the matched URL, or any identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from .config import PROJECT_ROOT, ChainPreset, chain_preset, contract_address, private_key

ARTIFACT = PROJECT_ROOT / "contracts" / "HashAnchor.json"


def artifact() -> dict[str, Any]:
    """The committed compile output, so no Solidity toolchain is needed at runtime."""
    return json.loads(ARTIFACT.read_text())


def connect(network: str | None = None) -> tuple[Web3, ChainPreset]:
    """Connect to the first responsive RPC for the network."""
    preset = chain_preset(network)
    failures: list[str] = []
    for url in preset.rpc_urls:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 30}))
        # Polygon is proof-of-authority: its blocks carry a 105-byte extraData
        # field that stock web3.py rejects when reading any block. Without this
        # middleware, fee estimation (and therefore every write) fails.
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        try:
            if w3.is_connected() and w3.eth.chain_id == preset.chain_id:
                return w3, preset
            failures.append(f"{url}: wrong chain id or not connected")
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            failures.append(f"{url}: {type(exc).__name__}")
    raise RuntimeError(
        f"No reachable RPC for {preset.name}. Tried:\n    "
        + "\n    ".join(failures)
        + "\n  Set FACEBLOCK_RPC_URL to override."
    )


def account() -> Account:
    return Account.from_key(private_key())


def _fees(w3: Web3) -> dict[str, int]:
    """Build EIP-1559 fee fields, falling back to legacy pricing if unsupported."""
    block = w3.eth.get_block("latest")
    base = block.get("baseFeePerGas")
    if base is None:
        return {"gasPrice": w3.eth.gas_price}
    try:
        priority = w3.eth.max_priority_fee
    except Exception:  # noqa: BLE001 - some public RPCs omit this method
        priority = w3.to_wei(30, "gwei")
    # Polygon rejects very low priority fees; keep a sane floor.
    priority = max(priority, w3.to_wei(25, "gwei"))
    return {"maxPriorityFeePerGas": priority, "maxFeePerGas": base * 2 + priority}


def _tx_params(w3: Web3, preset: ChainPreset, acct: Account) -> dict:
    """Base fields every transaction needs.

    These must be passed *into* `build_transaction` so web3 estimates gas
    against the correct sender - adding them afterwards leaves the estimate
    based on a default account and can under-fund the transaction.
    """
    return {
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": preset.chain_id,
        **_fees(w3),
    }


def _build(builder, w3: Web3, preset: ChainPreset, acct: Account) -> dict:
    """Build a transaction, translating the common funding failure into advice.

    Gas estimation is what actually rejects an underfunded account, and the raw
    RPC error ("insufficient funds for transfer") gives no hint about faucets.
    """
    try:
        return builder.build_transaction(_tx_params(w3, preset, acct))
    except Exception as exc:  # noqa: BLE001 - re-raised with context below
        if "insufficient funds" in str(exc).lower():
            balance = w3.from_wei(w3.eth.get_balance(acct.address), "ether")
            raise RuntimeError(
                f"{acct.address} has {balance} {preset.currency} on {preset.name} - "
                f"not enough to cover gas. Fund it at {preset.faucet}"
            ) from exc
        raise


def _hex(value) -> str:
    """Normalise a hash to 0x-prefixed hex (web3 v7's .hex() omits the prefix)."""
    text = value.hex() if hasattr(value, "hex") else str(value)
    return text if text.startswith("0x") else "0x" + text


def _send(w3: Web3, acct: Account, tx: dict) -> dict:
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  tx submitted: {_hex(tx_hash)}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt["status"] != 1:
        raise RuntimeError(f"Transaction reverted: {_hex(tx_hash)}")
    return receipt


def balance_report(network: str | None = None) -> str:
    w3, preset = connect(network)
    acct = account()
    wei = w3.eth.get_balance(acct.address)
    return (
        f"{acct.address} holds {w3.from_wei(wei, 'ether')} {preset.currency} "
        f"on {preset.name}"
    )


def deploy(network: str | None = None) -> str:
    """Deploy HashAnchor and return its address."""
    w3, preset = connect(network)
    acct = account()
    art = artifact()

    contract = w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
    print(f"  deploying HashAnchor to {preset.name} from {acct.address} ...")
    tx = _build(contract.constructor(), w3, preset, acct)
    receipt = _send(w3, acct, tx)
    address = receipt["contractAddress"]
    print(f"  deployed at {address}")
    if preset.explorer:
        print(f"  {preset.explorer}/address/{address}")
    return address


def _contract(w3: Web3, address: str | None = None):
    art = artifact()
    return w3.eth.contract(
        address=Web3.to_checksum_address(address or contract_address()), abi=art["abi"]
    )


def anchor(digest_hex: str, network: str | None = None) -> dict[str, Any]:
    """Write a digest on chain and return a receipt summary."""
    w3, preset = connect(network)
    acct = account()
    contract = _contract(w3)
    digest = Web3.to_bytes(hexstr=digest_hex)
    if len(digest) != 32:
        raise ValueError(f"Digest must be 32 bytes, got {len(digest)}")

    if contract.functions.isAnchored(digest).call():
        # Re-anchoring is rejected on chain by design; surface the original.
        timestamp, submitter = contract.functions.get(digest).call()
        raise RuntimeError(
            f"Digest already anchored at block time {timestamp} by {submitter}. "
            "The on-chain record is immutable - this is expected on a re-run."
        )

    tx = _build(contract.functions.anchor(digest), w3, preset, acct)
    receipt = _send(w3, acct, tx)
    tx_hash = _hex(receipt["transactionHash"])
    return {
        "network": preset.name,
        "chain_id": preset.chain_id,
        "contract": contract.address,
        "tx_hash": tx_hash,
        "block_number": receipt["blockNumber"],
        "submitter": acct.address,
        "explorer_url": f"{preset.explorer}/tx/{tx_hash}" if preset.explorer else None,
    }


def lookup(digest_hex: str, network: str | None = None) -> dict[str, Any] | None:
    """Read an anchor back. Returns None when the digest was never anchored."""
    w3, _ = connect(network)
    contract = _contract(w3)
    digest = Web3.to_bytes(hexstr=digest_hex)
    if not contract.functions.isAnchored(digest).call():
        return None
    timestamp, submitter = contract.functions.get(digest).call()
    return {"timestamp": timestamp, "submitter": submitter, "contract": contract.address}
