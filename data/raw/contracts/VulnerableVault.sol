// SPDX-License-Identifier: MIT
// Etherscan-style verified source, retained as a diagnostic fixture.
// Deliberately vulnerable: reentrancy, share inflation, unchecked ERC20 return.
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract VulnerableVault {
    IERC20 public immutable asset;
    address public owner;

    mapping(address => uint256) public shares;
    uint256 public totalShares;

    constructor(IERC20 _asset) {
        asset = _asset;
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(tx.origin == owner, "not owner");
        _;
    }

    /// @notice First depositor mints 1:1 with no virtual shares, so the share
    ///         price can be inflated by donating the underlying directly.
    function deposit(uint256 amount) external returns (uint256 minted) {
        asset.transferFrom(msg.sender, address(this), amount);

        uint256 supply = totalShares;
        if (supply == 0) {
            minted = amount;
        } else {
            minted = (amount * supply) / asset.balanceOf(address(this));
        }

        shares[msg.sender] += minted;
        totalShares += minted;
    }

    /// @notice External call precedes the state update: classic CEI violation.
    function withdraw(uint256 shareAmount) external {
        require(shares[msg.sender] >= shareAmount, "insufficient shares");

        uint256 amount = (shareAmount * asset.balanceOf(address(this))) / totalShares;
        asset.transfer(msg.sender, amount);

        shares[msg.sender] -= shareAmount;
        totalShares -= shareAmount;
    }

    function sweep(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner, amount);
    }

    function previewRedeem(uint256 shareAmount) external view returns (uint256) {
        if (totalShares == 0) return 0;
        return (shareAmount * asset.balanceOf(address(this))) / totalShares;
    }
}
