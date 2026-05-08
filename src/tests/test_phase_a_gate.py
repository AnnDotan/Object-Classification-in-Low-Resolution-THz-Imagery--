"""Phase A alignment-gate test (US-002).

Exercises evaluate_gate() across all 4 active (model, dataset) pairs at
each of the three bands (green, yellow, red) plus the two boundary
transitions, and confirms TransNeXt pairs raise KeyError.

Run: ``python -m src.tests.test_phase_a_gate``
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.phase_a_gate import (
    GREEN_TOLERANCE,
    PAPER_BASELINES,
    YELLOW_TOLERANCE,
    evaluate_gate,
    gate_explanation,
)


def _check_all_pairs_three_bands() -> None:
    """For each (model, dataset), green at paper, yellow at paper-5pp, red at paper-10pp."""
    for (model, dataset), paper in PAPER_BASELINES.items():
        green_acc = paper                  # exactly at paper → green
        yellow_acc = paper - 0.05          # −5pp → yellow (between −3 and −7)
        red_acc = paper - 0.10             # −10pp → red

        assert evaluate_gate(model, dataset, green_acc) == "green", (
            f"{model}/{dataset}: val_acc={green_acc} should be green at paper={paper}"
        )
        assert evaluate_gate(model, dataset, yellow_acc) == "yellow", (
            f"{model}/{dataset}: val_acc={yellow_acc} should be yellow at paper={paper}"
        )
        assert evaluate_gate(model, dataset, red_acc) == "red", (
            f"{model}/{dataset}: val_acc={red_acc} should be red at paper={paper}"
        )

    print("OK [bands] — all 4 (model,dataset) pairs hit green/yellow/red at the right offsets.")


def _check_band_boundaries() -> None:
    """Boundary cases: exactly at -3pp is green, just below is yellow; exactly at -7pp is yellow, just below is red."""
    pair = ("resnet50", "cifar10")
    paper = PAPER_BASELINES[pair]

    # Green / yellow boundary at paper − 3pp.
    assert evaluate_gate(*pair, paper - GREEN_TOLERANCE) == "green", (
        "exactly at paper − 3pp should still be green (closed boundary)"
    )
    # Just below the green boundary by 1e-6 should fall to yellow.
    assert evaluate_gate(*pair, paper - GREEN_TOLERANCE - 1e-6) == "yellow", (
        "1e-6 below paper − 3pp should be yellow"
    )

    # Yellow / red boundary at paper − 7pp.
    assert evaluate_gate(*pair, paper - YELLOW_TOLERANCE) == "yellow", (
        "exactly at paper − 7pp should still be yellow (closed boundary)"
    )
    assert evaluate_gate(*pair, paper - YELLOW_TOLERANCE - 1e-6) == "red", (
        "1e-6 below paper − 7pp should be red"
    )

    print("OK [boundaries] — closed-on-the-high-side boundary semantics confirmed.")


def _check_above_paper_is_green() -> None:
    """val_acc above the paper number is still green — not a special band."""
    pair = ("densenet121", "mnist")
    paper = PAPER_BASELINES[pair]
    # MNIST already saturated; impossible to exceed 1.0 but pin the contract anyway.
    assert evaluate_gate(*pair, min(paper + 0.005, 1.0)) == "green"
    assert evaluate_gate(*pair, 1.0) == "green"
    print("OK [above-paper] — val_acc >= paper stays green.")


def _check_unknown_pair_raises() -> None:
    """TransNeXt pairs are intentionally not in PAPER_BASELINES (deferred per PRD)."""
    for pair in [("transnext_small", "cifar10"), ("transnext_base", "mnist"),
                 ("resnet50", "imagenet"), ("vit_b16", "cifar10")]:
        try:
            evaluate_gate(*pair, 0.5)
        except KeyError as e:
            assert str(pair[0]) in str(e) or "no Phase A baseline" in str(e), (
                f"KeyError lacks pair name: {e}"
            )
            continue
        raise AssertionError(f"expected KeyError for unknown pair {pair}")
    print("OK [unknown-pair] — non-registered (model,dataset) raises KeyError.")


def _check_explanation_string_shape() -> None:
    """gate_explanation contains the band, val_acc, paper number, and delta in pp."""
    pair = ("resnet50", "cifar10")
    paper = PAPER_BASELINES[pair]
    msg = gate_explanation(*pair, paper - 0.05)  # yellow
    assert "YELLOW" in msg
    assert f"{paper:.3f}" in msg
    assert "-5.00pp" in msg, f"explanation missing delta: {msg}"
    print("OK [explanation] — gate_explanation includes band + paper + delta.")


def main() -> int:
    _check_all_pairs_three_bands()
    _check_band_boundaries()
    _check_above_paper_is_green()
    _check_unknown_pair_raises()
    _check_explanation_string_shape()
    return 0


if __name__ == "__main__":
    sys.exit(main())
