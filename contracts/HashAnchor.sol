// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title  HashAnchor
/// @notice Anchors SHA-256 digests of off-chain FaceBlock match records.
///
/// @dev    DESIGN NOTE - why only a digest is stored on chain:
///         A public ledger is immutable and world-readable. Writing a
///         face-to-identity claim onto it directly would publish an
///         irrevocable, potentially WRONG accusation of identity about a real
///         person, with no way to correct or erase it. So the match record
///         (photo hash, embedding hash, matched URL, confidence, timestamp)
///         stays in off-chain storage the operator controls, and only its
///         SHA-256 digest is anchored here. Anyone holding the off-chain
///         record can recompute the digest and check it against this contract:
///         that yields a tamper-EVIDENT, independently verifiable record
///         without publishing anyone's personal data permanently.
contract HashAnchor {
    struct Anchor {
        uint64 timestamp; // block time at which the digest was anchored
        address submitter; // account that anchored it
    }

    mapping(bytes32 => Anchor) private _anchors;

    event Anchored(bytes32 indexed digest, address indexed submitter, uint64 timestamp);

    error AlreadyAnchored(bytes32 digest);
    error NotAnchored(bytes32 digest);

    /// @notice Anchor a SHA-256 digest of a match record.
    /// @dev    Reverts if the digest is already anchored, so an existing
    ///         timestamp can never be silently overwritten - that immutability
    ///         is the whole point of the anchor.
    function anchor(bytes32 digest) external {
        if (_anchors[digest].timestamp != 0) revert AlreadyAnchored(digest);
        uint64 ts = uint64(block.timestamp);
        _anchors[digest] = Anchor({timestamp: ts, submitter: msg.sender});
        emit Anchored(digest, msg.sender, ts);
    }

    /// @notice Look up an anchored digest. Reverts if it was never anchored.
    function get(bytes32 digest) external view returns (uint64 timestamp, address submitter) {
        Anchor memory a = _anchors[digest];
        if (a.timestamp == 0) revert NotAnchored(digest);
        return (a.timestamp, a.submitter);
    }

    /// @notice Non-reverting existence check.
    function isAnchored(bytes32 digest) external view returns (bool) {
        return _anchors[digest].timestamp != 0;
    }
}
