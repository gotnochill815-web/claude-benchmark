#!/usr/bin/env python3
"""Collect results and actual usage from submitted Claude batches."""

import argparse
import json
from pathlib import Path

import anthropic

from common import MANIFEST_DIR, OUTPUT_DIR
from langfuse_tracking import create_langfuse, record_observation, flush_langfuse


def dump_model(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="Collect one run; default: all submitted runs")
    args = parser.parse_args()

    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is missing. No API calls made.")

    manifests = sorted(MANIFEST_DIR.glob("*.json"))
    if args.run_id:
        manifests = [p for p in manifests if p.stem == args.run_id]
        if not manifests:
            raise SystemExit(f"No manifest found for run ID: {args.run_id}")

    lf = create_langfuse()
    client = anthropic.Anthropic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collected_runs = 0

    for path in manifests:
        manifest = json.loads(path.read_text())
        if not manifest.get("submitted") or not manifest.get("batch_id"):
            continue

        run_id = manifest["run_id"]
        batch_id = manifest["batch_id"]
        print(f"\nRun: {run_id} | Batch: {batch_id}")

        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        manifest["batch_processing_status"] = status

        counts = getattr(batch, "request_counts", None)
        if counts:
            manifest["batch_request_counts"] = dump_model(counts)

        if status != "ended":
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print("Batch is not finished:", status)
            continue

        raw_path = OUTPUT_DIR / f"{run_id}_raw.jsonl"
        results_by_id = {}

        for item in client.messages.batches.results(batch_id):
            result = item.result
            record = {
                "custom_id": item.custom_id,
                "result_type": getattr(result, "type", "unknown"),
                "raw_result": dump_model(result),
            }

            if getattr(result, "type", None) == "succeeded":
                message = result.message
                text_blocks = [
                    block.text for block in message.content
                    if getattr(block, "type", None) == "text"
                ]
                usage = getattr(message, "usage", None)
                record["text"] = "\n".join(text_blocks)
                record["usage"] = dump_model(usage) if usage else {}
                record["stop_reason"] = getattr(message, "stop_reason", None)

            results_by_id[item.custom_id] = record

        with raw_path.open("w", encoding="utf-8") as f:
            for record in results_by_id.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for request in manifest.get("requests", []):
            result = results_by_id.get(request["custom_id"])
            if result is None:
                request["status"] = "missing_result"
                result = {
                    "result_type": "missing_result",
                    "text": None,
                    "usage": {},
                }
            else:
                request["status"] = result["result_type"]

            usage = result.get("usage") or {}
            usage_details = {}
            for source, dest in (
                ("input_tokens", "input"),
                ("output_tokens", "output"),
                ("cache_read_input_tokens", "cache_read_input"),
                ("cache_creation_input_tokens", "cache_creation_input"),
            ):
                value = usage.get(source)
                if isinstance(value, (int, float)):
                    usage_details[dest] = int(value)

            failed = result.get("result_type") != "succeeded"
            record_observation(
                lf,
                name="claude-logp-request",
                input_data={
                    "run_id": run_id,
                    "custom_id": request["custom_id"],
                    "target_logp": request.get("target_logp"),
                    "shots": manifest.get("shots"),
                    "strategy": manifest.get("strategy"),
                },
                output_data={
                    "result_type": result.get("result_type"),
                    "text": result.get("text"),
                    "usage": usage,
                },
                metadata={
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "custom_id": request["custom_id"],
                    "strategy": manifest.get("strategy"),
                    "shots": manifest.get("shots"),
                    "model": manifest.get("model"),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                },
                level="ERROR" if failed else None,
                status_message=result.get("result_type") if failed else None,
                model=manifest.get("model"),
                usage_details=usage_details or None,
            )

        manifest["results_file"] = str(raw_path)
        manifest["collected"] = True
        manifest["collected_result_count"] = len(results_by_id)
        path.write_text(json.dumps(manifest, indent=2) + "\n")

        print("Batch status:", status)
        print("Results collected:", len(results_by_id))
        print("Raw results:", raw_path)
        collected_runs += 1

    flush_langfuse(lf)
    print(f"\nCollection finished. Batches collected: {collected_runs}")


if __name__ == "__main__":
    main()
