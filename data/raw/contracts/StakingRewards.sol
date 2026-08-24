// SPDX-License-Identifier: MIT
// Diagnostic fixture: push distribution, unbounded loop, unchecked arithmetic.
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract StakingRewards {
    IERC20 public stakingToken;
    IERC20 public rewardToken;

    address[] public stakers;
    mapping(address => uint256) public staked;
    mapping(address => uint256) public released;
    mapping(address => uint256) public vested;

    constructor(IERC20 _staking, IERC20 _reward) {
        stakingToken = _staking;
        rewardToken = _reward;
    }

    /// @notice Credits the requested amount rather than the observed balance
    ///         delta, so fee-on-transfer tokens over-credit the depositor.
    function stake(uint256 amount) external {
        stakingToken.transferFrom(msg.sender, address(this), amount);
        if (staked[msg.sender] == 0) {
            stakers.push(msg.sender);
        }
        staked[msg.sender] += amount;
    }

    /// @notice Unbounded loop over every staker: griefable to past the gas limit.
    function distributeRewards(uint256 totalReward) external {
        uint256 n = stakers.length;
        for (uint256 i = 0; i < n; i++) {
            address account = stakers[i];
            uint256 share = (totalReward * staked[account]) / totalStaked();
            rewardToken.transfer(account, share);
        }
    }

    /// @notice Unchecked subtraction can wrap when released exceeds vested.
    function releasable(address account) public view returns (uint256) {
        unchecked {
            return vested[account] - released[account];
        }
    }

    function totalStaked() public view returns (uint256 total) {
        for (uint256 i = 0; i < stakers.length; i++) {
            total += staked[stakers[i]];
        }
    }

    /// @notice Randomness derived from block fields the builder controls.
    function luckyDraw() external view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp, block.prevrandao, msg.sender))) % 100;
    }
}
