#!/usr/bin/env python3
"""Evaluate collected Claude molecular LogP benchmark results."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from common import (
    MANIFEST_DIR,
    OUTPUT_DIR,
    TEST_PATH,
    canonicalize,
    calc_logp,
    extract_smiles,
)


def load_canonical_set(path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"No smiles column in {path}")

    return {
        value for smi in df["smiles"].dropna()
        if (value := canonicalize(str(smi))) is not None
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", help="Evaluate one run; default: all collected runs"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve manifests before accessing one, so an empty directory
    # produces a useful error instead of StopIteration.
    manifests = sorted(MANIFEST_DIR.glob("*.json"))
    if args.run_id:
        manifests = [p for p in manifests if p.stem == args.run_id]
        if not manifests:
            raise SystemExit(f"No manifest found for run ID: {args.run_id}")
    elif not manifests:
        raise SystemExit(
            f"No manifests found in {MANIFEST_DIR}. Prepare a benchmark first."
        )

    # Novelty is measured against the training set, as in the OpenAI baseline.
    first_manifest = json.loads(manifests[0].read_text())
    train_path = Path(first_manifest["train_path"])
    train_set = load_canonical_set(train_path)

    rows = []

    for path in manifests:
        manifest = json.loads(path.read_text())
        raw_file = manifest.get("results_file")
        if not manifest.get("collected") or not raw_file:
            continue

        raw_path = Path(raw_file)
        if not raw_path.is_file():
            print("Skipping missing results file:", raw_path)
            continue

        raw_by_id = {}
        with raw_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    raw_by_id[record["custom_id"]] = record

        for request in manifest.get("requests", []):
            raw = raw_by_id.get(request["custom_id"], {})
            output = raw.get("text")
            smiles = extract_smiles(output) if output else None
            valid = smiles is not None
            predicted_logp = calc_logp(smiles) if valid else None
            target = float(request["target_logp"])
            examples = request.get("prompt_example_smiles", [])
            example_set = {
                value for smi in examples
                if (value := canonicalize(smi)) is not None
            }

            usage = raw.get("usage") or {}
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_create = usage.get("cache_creation_input_tokens", 0) or 0

            rows.append({
                "run_id": manifest["run_id"],
                "strategy": manifest["strategy"],
                "shots": manifest["shots"],
                "model": manifest["model"],
                "custom_id": request["custom_id"],
                "target_logp": target,
                "raw_output": output,
                "smiles": smiles,
                "valid": valid,
                "heavy_atom_5plus": (
                    Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() >= 5
                    if valid else None
                ),
                "calculated_logp": predicted_logp,
                "absolute_logp_error": (
                    abs(predicted_logp - target)
                    if predicted_logp is not None else None
                ),
                "is_novel": (
                    smiles not in train_set if valid else None
                ),
                "prompt_copy": (
                    smiles in example_set if valid and manifest["shots"] > 0
                    else (False if valid and manifest["shots"] == 0 else None)
                ),
                "result_type": raw.get("result_type", "missing_result"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_create,
                "stop_reason": raw.get("stop_reason"),
            })

    if not rows:
        raise SystemExit(
            "No collected results found. Collect completed batches first."
        )

    detail = pd.DataFrame(rows)
    detail["valid_num"] = detail["valid"].astype(int)

    summaries = []
    for (run_id, strategy, shots, model, target_logp), group in detail.groupby(
        ["run_id", "strategy", "shots", "model", "target_logp"], dropna=False
    ):
        valid = group[group["valid"]]
        valid_smiles = valid["smiles"].dropna()
        unique_count = valid_smiles.nunique()
        n_total = len(group)
        n_valid = len(valid)
        novelty_values = valid["is_novel"].dropna()
        copy_values = valid["prompt_copy"].dropna()
        errors = valid["absolute_logp_error"].dropna()

        if shots > 0 and len(copy_values) and copy_values.mean() > 0.10:
            print(
                f"WARNING: exact prompt-copy rate exceeds 10%: "
                f"{run_id}, target LogP={target_logp}, "
                f"rate={copy_values.mean():.1%}"
            )

        heavy_values = group["heavy_atom_5plus"].dropna()

        summaries.append({
            "run_id": run_id,
            "strategy": strategy,
            "shots": shots,
            "model": model,
            "target_logp": target_logp,
            "heavy_atom_5plus_rate": (
                heavy_values.mean() if len(heavy_values) else None
            ),
            "n_requests": n_total,
            "n_results": int((group["result_type"] != "missing_result").sum()),
            "valid_count": n_valid,
            "validity": n_valid / n_total if n_total else None,
            "unique_valid_molecules": unique_count,
            "uniqueness": unique_count / n_valid if n_valid else None,
            "novelty": novelty_values.mean() if len(novelty_values) else None,
            "prompt_copy_rate": (
                copy_values.mean() if shots > 0 and len(copy_values) else None
            ),
            "mean_absolute_logp_error": errors.mean() if len(errors) else None,
            "input_tokens": pd.to_numeric(
                group["input_tokens"], errors="coerce"
            ).sum(min_count=1),
            "output_tokens": pd.to_numeric(
                group["output_tokens"], errors="coerce"
            ).sum(min_count=1),
            "cache_read_input_tokens": group[
                "cache_read_input_tokens"
            ].sum(),
            "cache_creation_input_tokens": group[
                "cache_creation_input_tokens"
            ].sum(),
        })

    detail_path = OUTPUT_DIR / "claude_logp_per_generation.csv"
    summary_path = OUTPUT_DIR / "claude_logp_summary.csv"

    detail.drop(columns=["valid_num"]).to_csv(detail_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print("Per-generation results:", detail_path)
    print("Summary:", summary_path)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
