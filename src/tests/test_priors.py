"""Validate the per-model hparam priors files and the loader rejection paths.

Two contracts:
  (a) Every shipped priors file (resnet50, densenet121, transnext_tiny) loads
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
    MAX_LOGUNIFORM_RATIO,
    PriorsValidationError,
    SUPPORTED_MODELS,
    load_priors,
    priors_file_hash,
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


def _check_decade_width_invariant() -> None:
    """US-015: every shipped loguniform band must satisfy ±1 decade from anchor
    (`high/low ≤ MAX_LOGUNIFORM_RATIO`); over-wide bands must be rejected."""
    # All shipped priors validate cleanly under the new invariant.
    for model in SUPPORTED_MODELS:
        load_priors(model)  # raises if any band is too wide
    # Construct a synthetic too-wide loguniform and confirm it is rejected.
    base = json.loads(
        (tune_all.PRIORS_DIR / "resnet50.json").read_text(encoding="utf-8")
    )
    too_wide = deepcopy(base)
    # 1e-5 -> 1e-1 is 10000x = 4 decades, well beyond the project cap.
    too_wide["hparams"]["head_lr"]["low"] = 1e-5
    too_wide["hparams"]["head_lr"]["high"] = 1e-1
    _expect_errors(too_wide, "loguniform band too wide")
    # An exactly-at-the-limit band is accepted.
    at_limit = deepcopy(base)
    at_limit["hparams"]["head_lr"]["low"] = 1e-3
    at_limit["hparams"]["head_lr"]["high"] = 1e-1
    errors = validate_priors(at_limit)
    assert errors == [], f"at-limit band wrongly rejected: {errors}"
    print(f"OK [decade-width] — MAX_LOGUNIFORM_RATIO={MAX_LOGUNIFORM_RATIO} enforced; "
          f"all {len(SUPPORTED_MODELS)} shipped priors comply.")


def _check_priors_file_hash_stability() -> None:
    """priors_file_hash must be (a) hex-encoded SHA-256 of the file bytes,
    (b) stable across calls, (c) different across model files,
    (d) raises FileNotFoundError for unknown models."""
    import hashlib

    digests: dict[str, str] = {}
    for model in SUPPORTED_MODELS:
        h1 = priors_file_hash(model)
        h2 = priors_file_hash(model)
        assert h1 == h2, f"{model}: digest not stable — {h1} != {h2}"
        assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1), \
            f"{model}: not a hex SHA-256 digest: {h1}"
        # Verify against an independent recomputation.
        path = tune_all.PRIORS_DIR / f"{model}.json"
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert h1 == expected, f"{model}: digest != sha256 of file bytes"
        digests[model] = h1
    # Different files must produce different hashes (sanity check).
    assert len(set(digests.values())) == len(digests), \
        f"priors files share digests (suspicious): {digests}"

    try:
        priors_file_hash("nonexistent_model_xyz")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("priors_file_hash did not raise for unknown model")
    print(f"OK [hash] — priors_file_hash stable for {len(SUPPORTED_MODELS)} models; "
          f"unknown model raises FileNotFoundError.")


def _check_priors_sources_md_present() -> None:
    """US-015: PRIORS_SOURCES.md must exist alongside the per-model JSONs and
    cite the paper file for every supported model."""
    md_path = tune_all.PRIORS_DIR / "PRIORS_SOURCES.md"
    assert md_path.exists(), f"missing {md_path}"
    body = md_path.read_text(encoding="utf-8")
    # Every shipped model must be referenced.
    for model in SUPPORTED_MODELS:
        assert model in body, f"PRIORS_SOURCES.md does not mention {model!r}"
    # The decade-width convention must be spelled out so future contributors
    # discover it before authoring a new prior.
    assert "MAX_LOGUNIFORM_RATIO" in body or "high / low" in body, \
        "PRIORS_SOURCES.md must document the decade-width invariant"
    # Each shipped prior's primary paper file must be cited at least once.
    for paper in (
        "papers/TResNet.pdf",
        "papers/Densely Connected Convolutional Networks.pdf",
        "papers/TransNeXt.pdf",
    ):
        assert paper in body or paper.replace(" ", "%20") in body, \
            f"PRIORS_SOURCES.md missing citation for {paper!r}"
    print(f"OK [sources-md] — PRIORS_SOURCES.md present, cites all "
          f"{len(SUPPORTED_MODELS)} models + 3 papers.")


def main() -> int:
    _check_shipped_priors_validate()
    _check_malformed_rejected()
    _check_load_priors_raises_on_disk()
    _check_decade_width_invariant()
    _check_priors_file_hash_stability()
    _check_priors_sources_md_present()
    return 0


if __name__ == "__main__":
    sys.exit(main())
