// SPDX-License-Identifier: MIT
// Diagnostic fixture: bridge release path with replay and ecrecover defects.
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
}

contract NaiveBridge {
    address public signer;
    IERC20 public token;
    mapping(uint256 => bool) public usedNonce;

    event Released(address indexed to, uint256 amount, uint256 nonce);

    constructor(address _signer, IERC20 _token) {
        signer = _signer;
        token = _token;
    }

    /// @notice Digest omits block.chainid and address(this): replayable on any
    ///         chain where this bridge is deployed with the same signer set.
    function messageHash(address to, uint256 amount, uint256 nonce) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(to, amount, nonce));
    }

    /// @notice ecrecover result is not checked against address(0) and s is not
    ///         constrained to the lower half order, so signatures are malleable.
    function release(address to, uint256 amount, uint256 nonce, uint8 v, bytes32 r, bytes32 s) external {
        require(!usedNonce[nonce], "nonce used");

        bytes32 digest = messageHash(to, amount, nonce);
        address recovered = ecrecover(digest, v, r, s);
        require(recovered == signer, "bad signature");

        usedNonce[nonce] = true;
        token.transfer(to, amount);
        emit Released(to, amount, nonce);
    }

    function updateSigner(address newSigner) external {
        signer = newSigner;
    }
}
