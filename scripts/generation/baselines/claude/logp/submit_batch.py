#!/usr/bin/env python3
"""Prepare or submit Claude molecular LogP benchmark batches."""

import argparse
import json
import os
from datetime import datetime, timezone

import anthropic

from common import (
    ALLOWED_SHOTS,
    MANIFEST_DIR,
    MODEL,
    N_PER_TARGET,
    OUTPUT_DIR,
    SEED,
    TARGETS,
    build_example_orders,
    build_user_content,
    load_train,
    safe_name,
    select_examples,
    SYSTEM_PROMPT,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=["zero", "random", "nearest"],
        required=True,
    )
    parser.add_argument(
        "--shots",
        type=int,
        choices=ALLOWED_SHOTS,
        required=True,
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the prepared JSONL as an Anthropic batch.",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.strategy == "zero" and args.shots != 0:
        raise SystemExit("Strategy 'zero' requires --shots 0.")
    if args.strategy != "zero" and args.shots == 0:
        raise SystemExit("Random/nearest require --shots 5, 10, 100, or 500.")

    train_df = load_train()
    orders = build_example_orders(train_df, seed=SEED)

    run_name = (
        f"{args.strategy}_{args.shots}shot_"
        f"{safe_name(args.model)}"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_name}_{stamp}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = OUTPUT_DIR / f"{run_id}.jsonl"
    manifest_path = MANIFEST_DIR / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "created_at_utc": stamp,
        "strategy": args.strategy,
        "shots": args.shots,
        "model": args.model,
        "seed": SEED,
        "n_per_target": N_PER_TARGET,
        "targets": TARGETS,
        "train_path": str(__import__("common").TRAIN_PATH),
        "requests_file": str(jsonl_path),
        "submitted": False,
        "batch_id": None,
        "requests": [],
    }

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for target_idx, target in enumerate(TARGETS):
            examples = select_examples(
                orders,
                args.strategy,
                target_idx,
                args.shots,
            )

            for repeat_idx in range(N_PER_TARGET):
                custom_id = (
                    f"t{target_idx:02d}_r{repeat_idx + 1:02d}"
                )
                user_content = build_user_content(
                    target,
                    examples,
                    args.shots,
                )

                request = {
                    "custom_id": custom_id,
                    "params": {
                        "model": args.model,
                        "max_tokens": args.max_tokens,
                        "temperature": 0.7,
                        "system": [
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT,
                                "cache_control": {
                                    "type": "ephemeral",
                                    "ttl": "1h",
                                },
                            }
                        ],
                        "messages": [
                            {
                                "role": "user",
                                "content": user_content,
                            }
                        ],
                    },
                }

                jf.write(json.dumps(request, ensure_ascii=False) + "\n")

                manifest["requests"].append({
                    "custom_id": custom_id,
                    "target_index": target_idx,
                    "target_logp": float(target),
                    "repeat": repeat_idx + 1,
                    "prompt_example_smiles": examples["smiles"].tolist(),
                    "prompt_example_logp": [
                        float(x) for x in examples["logp"].tolist()
                    ],
                    "status": "prepared",
                })

    manifest["requests_file"] = str(jsonl_path)

    if args.submit:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. No batch was submitted."
            )

        client = anthropic.Anthropic()
        with jsonl_path.open("rb") as jf:
            batch = client.messages.batches.create(
                requests=[
                    json.loads(line)
                    for line in jf
                    if line.strip()
                ]
            )

        manifest["submitted"] = True
        manifest["batch_id"] = batch.id
        manifest["batch_processing_status"] = batch.processing_status

    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(f"Run ID:          {run_id}")
    print(f"Strategy:        {args.strategy}")
    print(f"Shots:           {args.shots}")
    print(f"Model:           {args.model}")
    print(f"Requests:        {len(manifest['requests'])}")
    print(f"JSONL:           {jsonl_path}")
    print(f"Manifest:        {manifest_path}")
    print(f"Submitted:       {manifest['submitted']}")

    if manifest["submitted"]:
        print(f"Batch ID:        {manifest['batch_id']}")
    else:
        print("Dry run only. No API request was sent.")


if __name__ == "__main__":
    main()
