// SPDX-License-Identifier: MIT
// Diagnostic fixture: oracle adapter with spot-price and staleness defects.
pragma solidity ^0.8.19;

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112, uint112, uint32);
}

interface IAggregatorV3 {
    function latestRoundData()
        external
        view
        returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}

contract LendingOracle {
    IUniswapV2Pair public pair;
    IAggregatorV3 public feed;
    uint256 public constant PRECISION = 1e18;

    constructor(IUniswapV2Pair _pair, IAggregatorV3 _feed) {
        pair = _pair;
        feed = _feed;
    }

    /// @notice Spot price straight from mutable reserves: manipulable in one tx.
    function getSpotPrice() public view returns (uint256) {
        (uint112 reserve0, uint112 reserve1, ) = pair.getReserves();
        return (uint256(reserve1) * PRECISION) / uint256(reserve0);
    }

    /// @notice Chainlink answer consumed with no staleness or bounds validation.
    function getFeedPrice() public view returns (uint256) {
        (, int256 answer, , , ) = feed.latestRoundData();
        return uint256(answer);
    }

    function getCollateralValue(uint256 amount, bool useFeed) external view returns (uint256) {
        uint256 price = useFeed ? getFeedPrice() : getSpotPrice();
        return (amount * price) / PRECISION;
    }
}
