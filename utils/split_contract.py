"""Strict split validation used by the clean ICRA training protocol."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set


def _resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(os.path.expanduser(str(path_value)))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def read_split(path: Path) -> List[str]:
    """Read a K-Radar split and reject comments, malformed rows, and duplicates."""
    rows: List[str] = []
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            row = raw_line.strip()
            if not row:
                raise ValueError(f"blank split row: {path}:{line_number}")
            if row.startswith("#"):
                raise ValueError(f"commented split row is forbidden: {path}:{line_number}")
            fields = row.split(",")
            if len(fields) != 2 or not fields[0] or not fields[1].endswith(".txt"):
                raise ValueError(f"malformed split row: {path}:{line_number}: {row!r}")
            if row in seen:
                raise ValueError(f"duplicate split row: {path}:{line_number}: {row!r}")
            seen.add(row)
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_disjoint(named_sets: Dict[str, Set[str]]) -> None:
    names = list(named_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = named_sets[left_name] & named_sets[right_name]
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(
                    f"split overlap {left_name}/{right_name}: {len(overlap)} rows; examples={examples}"
                )


def verify_split_contract(cfg, project_root: str | Path | None = None) -> dict:
    """Validate clean internal train/val and untouched 3D-LRF official test splits."""
    guard = cfg.DATASET.get("SPLIT_GUARD", {})
    if not bool(guard.get("ENABLED", False)):
        return {"enabled": False}

    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    project_root = Path(project_root).resolve()

    path_cfg = cfg.DATASET.PATH_SPLIT
    required = ("train", "val", "test", "official_train", "official_test")
    missing = [name for name in required if not path_cfg.get(name)]
    if missing:
        raise KeyError(f"DATASET.PATH_SPLIT is missing required entries: {missing}")

    paths = {name: _resolve_path(path_cfg[name], project_root) for name in required}
    rows = {name: read_split(path) for name, path in paths.items()}
    row_sets = {name: set(values) for name, values in rows.items()}
    protocol_phase = str(guard.get("PROTOCOL_PHASE", "stage2_selection")).strip().lower()
    supported_phases = {"stage2_selection", "stage2_full_train_refit"}
    if protocol_phase not in supported_phases:
        raise ValueError(
            f"unsupported DATASET.SPLIT_GUARD.PROTOCOL_PHASE={protocol_phase!r}; "
            f"expected one of {sorted(supported_phases)}"
        )

    _require_disjoint(
        {"official_train": row_sets["official_train"], "official_test": row_sets["official_test"]}
    )
    official_train_set = row_sets["official_train"]
    if rows["test"] != rows["official_test"]:
        raise ValueError("runtime test split is not byte-order equivalent to official_test")

    if protocol_phase == "stage2_selection":
        _require_disjoint({name: row_sets[name] for name in ("train", "val", "test")})
        internal_union = row_sets["train"] | row_sets["val"]
        if internal_union != official_train_set:
            missing_from_internal = sorted(official_train_set - internal_union)[:5]
            extra_in_internal = sorted(internal_union - official_train_set)[:5]
            raise ValueError(
                "internal train+val does not exactly reconstruct official train: "
                f"missing={len(official_train_set - internal_union)} {missing_from_internal}; "
                f"extra={len(internal_union - official_train_set)} {extra_in_internal}"
            )
    else:
        if rows["train"] != rows["official_train"]:
            raise ValueError(
                "stage2_full_train_refit requires runtime train to be byte-order equivalent "
                "to official_train"
            )
        _require_disjoint(
            {"train": row_sets["train"], "test": row_sets["test"]}
        )
        _require_disjoint(
            {"val": row_sets["val"], "test": row_sets["test"]}
        )
        if not row_sets["val"].issubset(official_train_set):
            raise ValueError("refit config val split is not a subset of official_train")
        if bool(cfg.VAL.get("IS_VALIDATE", True)):
            raise ValueError("stage2_full_train_refit must disable validation")
        if bool(cfg.VAL.get("RUN_CONDITIONAL_AT_END", True)):
            raise ValueError("stage2_full_train_refit must disable end-of-training test evaluation")

    expected_train_count = int(guard.get("OFFICIAL_TRAIN_COUNT", len(rows["official_train"])))
    expected_test_count = int(guard.get("OFFICIAL_TEST_COUNT", len(rows["official_test"])))
    if len(rows["official_train"]) != expected_train_count:
        raise ValueError(
            f"official train count {len(rows['official_train'])} != expected {expected_train_count}"
        )
    if len(rows["official_test"]) != expected_test_count:
        raise ValueError(
            f"official test count {len(rows['official_test'])} != expected {expected_test_count}"
        )

    expected_train_hash = str(guard.get("OFFICIAL_TRAIN_SHA256", "")).strip()
    expected_test_hash = str(guard.get("OFFICIAL_TEST_SHA256", "")).strip()
    train_hash = sha256_file(paths["official_train"])
    test_hash = sha256_file(paths["official_test"])
    if expected_train_hash and train_hash != expected_train_hash:
        raise ValueError(f"official train SHA-256 mismatch: {train_hash} != {expected_train_hash}")
    if expected_test_hash and test_hash != expected_test_hash:
        raise ValueError(f"official test SHA-256 mismatch: {test_hash} != {expected_test_hash}")

    train_sequences = {row.split(",", 1)[0] for row in rows["train"]}
    val_sequences = {row.split(",", 1)[0] for row in rows["val"]}
    sequence_overlap = train_sequences & val_sequences
    if protocol_phase == "stage2_selection" and sequence_overlap:
        raise ValueError(f"internal train/val sequence leakage: {sorted(sequence_overlap)}")

    summary = {
        "enabled": True,
        "protocol_phase": protocol_phase,
        "paths": {name: str(path) for name, path in paths.items()},
        "counts": {name: len(values) for name, values in rows.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "overlap_counts": {
            "train_val": len(row_sets["train"] & row_sets["val"]),
            "train_test": len(row_sets["train"] & row_sets["test"]),
            "val_test": len(row_sets["val"] & row_sets["test"]),
            "train_val_sequences": len(sequence_overlap),
        },
    }

    manifest_path = str(guard.get("MANIFEST_PATH", "")).strip()
    if manifest_path:
        expected_manifest = _resolve_path(manifest_path, project_root)
        with expected_manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("official_train_sha256") != train_hash:
            raise ValueError("split manifest official_train_sha256 does not match runtime split")
        if manifest.get("official_test_sha256") != test_hash:
            raise ValueError("split manifest official_test_sha256 does not match runtime split")

    # Local import avoids a cycle: Stage 1 verification reuses split parsing.
    from utils.stage1_contract import verify_stage1_contract
    summary['stage1'] = verify_stage1_contract(cfg, project_root, summary)
    return summary


def format_split_summary(summary: dict) -> str:
    if not summary.get("enabled", False):
        return "split guard disabled"
    counts = summary["counts"]
    hashes = summary["sha256"]
    overlaps = summary["overlap_counts"]
    return (
        "clean split contract: "
        f"train={counts['train']}, val={counts['val']}, test={counts['test']}; "
        f"overlap(train/val,train/test,val/test)="
        f"{overlaps['train_val']}/{overlaps['train_test']}/{overlaps['val_test']}; "
        f"official_sha256(train/test)={hashes['official_train']}/{hashes['official_test']}"
    )
