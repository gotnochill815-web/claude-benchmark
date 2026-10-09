#!/usr/bin/env python3
"""Submit an already-prepared Claude batch from an existing manifest."""

import argparse
import json
import os
from pathlib import Path

import anthropic

from common import MANIFEST_DIR
from langfuse_tracking import create_langfuse, record_observation, flush_langfuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit the batch.")
    args = parser.parse_args()

    manifest_path = MANIFEST_DIR / f"{args.run_id}.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    if manifest.get("submitted"):
        raise SystemExit(
            f"Already marked submitted. Batch ID: {manifest.get('batch_id')}"
        )

    requests_path = Path(manifest["requests_file"])
    if not requests_path.is_file():
        raise SystemExit(f"Request file not found: {requests_path}")

    with requests_path.open(encoding="utf-8") as f:
        requests = [json.loads(line) for line in f if line.strip()]

    expected = len(manifest["requests"])
    if len(requests) != expected:
        raise SystemExit(
            f"Request count mismatch: JSONL={len(requests)}, manifest={expected}"
        )

    print("Run ID:", manifest["run_id"])
    print("Model:", manifest["model"])
    print("Strategy:", manifest["strategy"])
    print("Shots:", manifest["shots"])
    print("Requests:", len(requests))

    if not args.submit:
        print("DRY RUN: no API request made.")
        print("Add --submit only when ready to incur API charges.")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is missing. Nothing submitted.")

    lf = create_langfuse()
    client = anthropic.Anthropic()
    try:
        batch = client.messages.batches.create(requests=requests)
    except Exception as exc:
        record_observation(
            lf,
            name="claude-logp-batch-submit",
            input_data={"run_id": manifest["run_id"], "request_count": len(requests)},
            output_data={"error_type": type(exc).__name__},
            metadata={
                "run_id": manifest["run_id"],
                "strategy": manifest.get("strategy"),
                "shots": manifest.get("shots"),
                "model": manifest.get("model"),
            },
            level="ERROR",
            status_message="Batch submission failed",
            model=manifest.get("model"),
        )
        flush_langfuse(lf)
        raise

    manifest["submitted"] = True
    manifest["batch_id"] = batch.id
    manifest["batch_processing_status"] = batch.processing_status
    manifest["batch_request_counts"] = (
        batch.request_counts.model_dump()
        if getattr(batch, "request_counts", None) else None
    )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("SUBMITTED")
    print("Batch ID:", batch.id)
    print("Processing status:", batch.processing_status)
    record_observation(
        lf,
        name="claude-logp-batch-submit",
        input_data={"run_id": manifest["run_id"], "request_count": len(requests)},
        output_data={
            "batch_id": batch.id,
            "processing_status": batch.processing_status,
        },
        metadata={
            "run_id": manifest["run_id"],
            "strategy": manifest.get("strategy"),
            "shots": manifest.get("shots"),
            "model": manifest.get("model"),
            "request_count": len(requests),
        },
        model=manifest.get("model"),
    )
    flush_langfuse(lf)
    print("Manifest updated:", manifest_path)


if __name__ == "__main__":
    main()
