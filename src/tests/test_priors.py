"""Validate the per-model hparam priors files and the loader rejection paths.

Two contracts:
  (a) Every shipped priors file (resnet50, densenet121, transnext_base) loads
      and validates cleanly.
  (b) The loader rejects malformed priors with a descriptive message —
      negative `low`, `low > high`, missing required field, missing
      citation, and unsupported distribution all surface as
      PriorsValidationError naming the offending field.

Run: ``python -m src.tests.test_priors``
"""
from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import tune_all
from tune_all import (
    PriorsValidationError,
    SUPPORTED_MODELS,
    load_priors,
    validate_priors,
)


def _check_shipped_priors_validate() -> None:
    for model in SUPPORTED_MODELS:
        data = load_priors(model)
        assert data["model"] == model, f"{model}: model field mismatch"
        for required in (
            "head_lr", "backbone_lr", "weight_decay",
            "label_smoothing", "warmup_epochs",
        ):
            assert required in data["hparams"], (
                f"{model}: missing required hparam {required}"
            )
            hp = data["hparams"][required]
            assert "citation" in hp and hp["citation"].strip(), (
                f"{model}.{required}: missing or empty citation"
            )
        print(f"OK [{model}] — priors validate, all 5 hparams cite a paper.")


def _expect_errors(data: dict, must_mention: str) -> None:
    errors = validate_priors(data)
    assert errors, f"expected validation to fail (missing: {must_mention!r})"
    blob = "\n".join(errors)
    assert must_mention in blob, (
        f"expected error mentioning {must_mention!r}, got:\n{blob}"
    )


def _check_malformed_rejected() -> None:
    base = json.loads(
        (tune_all.PRIORS_DIR / "resnet50.json").read_text(encoding="utf-8")
    )

    bad_neg = deepcopy(base)
    bad_neg["hparams"]["head_lr"]["low"] = -1.0
    _expect_errors(bad_neg, "loguniform requires strictly positive")

    bad_swap = deepcopy(base)
    bad_swap["hparams"]["backbone_lr"]["low"] = 1.0
    bad_swap["hparams"]["backbone_lr"]["high"] = 1e-5
    _expect_errors(bad_swap, "low (1.0) > high")

    bad_missing = deepcopy(base)
    del bad_missing["hparams"]["weight_decay"]
    _expect_errors(bad_missing, "hparams.weight_decay: required hparam is missing")

    bad_no_cite = deepcopy(base)
    bad_no_cite["hparams"]["label_smoothing"]["citation"] = ""
    _expect_errors(bad_no_cite, "label_smoothing.citation")

    bad_dist = deepcopy(base)
    bad_dist["hparams"]["warmup_epochs"]["distribution"] = "exponential"
    _expect_errors(bad_dist, "warmup_epochs.distribution")

    print("OK [malformed] — every malformed-priors variant rejected with a clear field path.")


def _check_load_priors_raises_on_disk() -> None:
    """End-to-end: write a malformed priors file to disk and load_priors must raise."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        bad = json.loads(
            (tune_all.PRIORS_DIR / "resnet50.json").read_text(encoding="utf-8")
        )
        bad["hparams"]["head_lr"]["low"] = 10.0
        bad["hparams"]["head_lr"]["high"] = 0.001
        target = td_path / "resnet50.json"
        target.write_text(json.dumps(bad), encoding="utf-8")

        original_dir = tune_all.PRIORS_DIR
        try:
            tune_all.PRIORS_DIR = td_path
            try:
                load_priors("resnet50")
            except PriorsValidationError as e:
                assert "head_lr" in str(e), f"error doesn't name head_lr: {e}"
                print("OK [disk] — load_priors raised PriorsValidationError on bad file.")
                return
            raise AssertionError("load_priors did not raise on malformed file")
        finally:
            tune_all.PRIORS_DIR = original_dir


def main() -> int:
    _check_shipped_priors_validate()
    _check_malformed_rejected()
    _check_load_priors_raises_on_disk()
    return 0


if __name__ == "__main__":
    sys.exit(main())
