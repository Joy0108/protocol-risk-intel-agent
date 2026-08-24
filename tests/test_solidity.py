from __future__ import annotations

from pria.config import RAW_DIR
from pria.solidity.ast_lite import _strip_for_scan, detect_patterns, extract_spans, summarise

VAULT = (RAW_DIR / "contracts" / "VulnerableVault.sol").read_text(encoding="utf-8")
ORACLE = (RAW_DIR / "contracts" / "LendingOracle.sol").read_text(encoding="utf-8")
BRIDGE = (RAW_DIR / "contracts" / "NaiveBridge.sol").read_text(encoding="utf-8")


def test_strip_for_scan_preserves_every_offset():
    source = 'contract A { string s = "a { b"; // } comment\n uint x; }'
    stripped = _strip_for_scan(source)
    assert len(stripped) == len(source)
    assert stripped.count("\n") == source.count("\n")
    # The brace inside the string literal and the one in the comment are gone.
    assert stripped.count("{") == 1 and stripped.count("}") == 1


def test_spans_are_declaration_aligned():
    spans = extract_spans(VAULT, "VulnerableVault.sol")
    by_name = {s.qualified_name: s for s in spans}
    assert "VulnerableVault" in by_name
    for name in ("VulnerableVault.deposit", "VulnerableVault.withdraw", "VulnerableVault.onlyOwner"):
        assert name in by_name, f"missing span {name}"
        span = by_name[name]
        assert span.start_line < span.end_line
        assert span.text.count("{") == span.text.count("}")

    assert by_name["VulnerableVault.onlyOwner"].kind == "modifier"
    assert by_name["VulnerableVault.withdraw"].kind == "function"


def test_interface_declarations_without_bodies_do_not_produce_spans():
    spans = extract_spans(VAULT, "VulnerableVault.sol")
    # IERC20 declares functions with no body; none of them should become a span.
    assert not [s for s in spans if s.contract == "IERC20" and s.kind == "function"]


def test_reentrancy_is_flagged_where_the_state_write_follows_the_call():
    spans = {s.qualified_name: s for s in extract_spans(VAULT, "VulnerableVault.sol")}
    rules = {f.rule_id for f in spans["VulnerableVault.withdraw"].findings}
    assert "cei-violation" in rules
    assert "SWC-107" in {f.swc for f in spans["VulnerableVault.withdraw"].findings}

    # deposit writes state after transferFrom too, but the important negative is
    # that a pure view function is not flagged for reentrancy.
    assert "cei-violation" not in {f.rule_id for f in spans["VulnerableVault.previewRedeem"].findings}


def test_tx_origin_authorisation_is_flagged():
    spans = {s.qualified_name: s for s in extract_spans(VAULT, "VulnerableVault.sol")}
    hits = spans["VulnerableVault.onlyOwner"].findings
    assert [h.swc for h in hits] == ["SWC-115"]


def test_oracle_patterns_are_flagged():
    spans = {s.qualified_name: s for s in extract_spans(ORACLE, "LendingOracle.sol")}
    assert "spot-price-reserves" in {f.rule_id for f in spans["LendingOracle.getSpotPrice"].findings}
    assert "oracle-staleness" in {f.rule_id for f in spans["LendingOracle.getFeedPrice"].findings}


def test_ecrecover_without_a_zero_address_check_is_flagged():
    spans = {s.qualified_name: s for s in extract_spans(BRIDGE, "NaiveBridge.sol")}
    assert "ecrecover-unchecked" in {f.rule_id for f in spans["NaiveBridge.release"].findings}


def test_a_guarded_ecrecover_is_not_flagged():
    """The rule must look at the surrounding lines, not just the call."""
    source = """
    contract Safe {
        function verify(bytes32 d, uint8 v, bytes32 r, bytes32 s) external view returns (bool) {
            address recovered = ecrecover(d, v, r, s);
            require(recovered != address(0), "bad sig");
            return recovered == signer;
        }
    }
    """
    hits = detect_patterns(source)
    assert "ecrecover-unchecked" not in {h.rule_id for h in hits}


def test_summarise_counts_every_rule():
    spans = extract_spans(VAULT, "VulnerableVault.sol")
    summary = summarise(spans)
    assert summary["callables"] >= 5
    assert summary["pattern_hits"] == sum(len(s.findings) for s in spans)
    assert "SWC-115" in summary["swc_set"]
