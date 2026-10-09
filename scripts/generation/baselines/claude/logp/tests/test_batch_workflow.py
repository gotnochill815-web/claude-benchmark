import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import submit_prepared
import collect_batch


def make_manifest(tmp_path, run_id="test-run", shots=0):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    requests_path = tmp_path / f"{run_id}.jsonl"

    requests = [
        {
            "custom_id": f"req-{i}",
            "params": {"model": "claude-3-5-haiku-latest"},
        }
        for i in range(2)
    ]
    requests_path.write_text(
        "".join(json.dumps(r) + "\n" for r in requests)
    )

    manifest = {
        "run_id": run_id,
        "model": "claude-3-5-haiku-latest",
        "strategy": "zero" if shots == 0 else "random",
        "shots": shots,
        "requests": requests,
        "requests_file": str(requests_path),
        "train_path": str(tmp_path / "train.csv"),
        "submitted": False,
        "batch_id": None,
    }
    manifest_path = manifest_dir / f"{run_id}.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_dir, manifest_path, requests_path


def test_submit_prepared_dry_run_never_creates_client(tmp_path, monkeypatch):
    manifest_dir, manifest_path, _ = make_manifest(tmp_path)
    monkeypatch.setattr(submit_prepared, "MANIFEST_DIR", manifest_dir)
    client = MagicMock()
    monkeypatch.setattr(submit_prepared.anthropic, "Anthropic", client)
    monkeypatch.setattr(
        sys, "argv", ["submit_prepared.py", "--run-id", "test-run"]
    )

    submit_prepared.main()

    client.assert_not_called()
    saved = json.loads(manifest_path.read_text())
    assert saved["submitted"] is False
    assert saved["batch_id"] is None


def test_submit_prepared_refuses_missing_api_key(tmp_path, monkeypatch):
    manifest_dir, manifest_path, _ = make_manifest(tmp_path)
    monkeypatch.setattr(submit_prepared, "MANIFEST_DIR", manifest_dir)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = MagicMock()
    monkeypatch.setattr(submit_prepared.anthropic, "Anthropic", client)
    monkeypatch.setattr(
        sys, "argv",
        ["submit_prepared.py", "--run-id", "test-run", "--submit"],
    )

    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is missing"):
        submit_prepared.main()

    client.assert_not_called()
    saved = json.loads(manifest_path.read_text())
    assert saved["submitted"] is False


def test_collect_skips_unsubmitted_manifest(tmp_path, monkeypatch):
    manifest_dir, manifest_path, _ = make_manifest(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(collect_batch, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(collect_batch, "OUTPUT_DIR", output_dir)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-test-key")
    client = MagicMock()
    monkeypatch.setattr(collect_batch.anthropic, "Anthropic", lambda: client)
    monkeypatch.setattr(
        sys, "argv", ["collect_batch.py", "--run-id", "test-run"]
    )
    monkeypatch.setattr(collect_batch, "create_langfuse", lambda: None)
    monkeypatch.setattr(collect_batch, "flush_langfuse", lambda *_: None)

    collect_batch.main()

    client.messages.batches.retrieve.assert_not_called()
    client.messages.batches.results.assert_not_called()
    saved = json.loads(manifest_path.read_text())
    assert saved["submitted"] is False


def test_submit_prepared_refuses_request_count_mismatch(tmp_path, monkeypatch):
    manifest_dir, manifest_path, requests_path = make_manifest(tmp_path)
    requests_path.write_text('{"custom_id":"only-one","params":{}}\n')
    monkeypatch.setattr(submit_prepared, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(
        sys, "argv", ["submit_prepared.py", "--run-id", "test-run"]
    )

    with pytest.raises(SystemExit, match="Request count mismatch"):
        submit_prepared.main()


def test_collect_completed_batch_records_success_and_missing_result(
    tmp_path, monkeypatch
):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    output_dir = tmp_path / "output"

    manifest = {
        "run_id": "completed-test",
        "model": "claude-3-5-haiku-latest",
        "strategy": "zero",
        "shots": 0,
        "submitted": True,
        "batch_id": "mock-batch-123",
        "requests": [
            {"custom_id": "req-1", "target_logp": 2.0},
            {"custom_id": "req-2", "target_logp": 3.0},
        ],
    }
    manifest_path = manifest_dir / "completed-test.json"
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(collect_batch, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(collect_batch, "OUTPUT_DIR", output_dir)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-test-key")
    monkeypatch.setattr(
        sys, "argv", ["collect_batch.py", "--run-id", "completed-test"]
    )
    monkeypatch.setattr(collect_batch, "create_langfuse", lambda: None)
    monkeypatch.setattr(collect_batch, "flush_langfuse", lambda *_: None)
    monkeypatch.setattr(collect_batch, "record_observation", lambda *a, **k: None)

    successful_result = SimpleNamespace(
        type="succeeded",
        message=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="CCO")],
            usage=SimpleNamespace(
                model_dump=lambda mode="json": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            ),
            stop_reason="end_turn",
        ),
        model_dump=lambda mode="json": {"type": "succeeded"},
    )
    result_item = SimpleNamespace(
        custom_id="req-1",
        result=successful_result,
    )

    client = MagicMock()
    client.messages.batches.retrieve.return_value = SimpleNamespace(
        processing_status="ended",
        request_counts=None,
    )
    client.messages.batches.results.return_value = [result_item]
    monkeypatch.setattr(
        collect_batch.anthropic, "Anthropic", lambda: client
    )

    collect_batch.main()

    saved = json.loads(manifest_path.read_text())
    assert saved["collected"] is True
    assert saved["collected_result_count"] == 1

    raw_path = Path(saved["results_file"])
    raw_records = [
        json.loads(line)
        for line in raw_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(raw_records) == 1
    assert raw_records[0]["custom_id"] == "req-1"
    assert raw_records[0]["text"] == "CCO"

    request_status = {
        request["custom_id"]: request["status"]
        for request in saved["requests"]
    }
    assert request_status["req-1"] == "succeeded"
    assert request_status["req-2"] == "missing_result"

    client.messages.batches.retrieve.assert_called_once_with(
        "mock-batch-123"
    )
    client.messages.batches.results.assert_called_once_with(
        "mock-batch-123"
    )


def test_evaluate_synthetic_results_writes_correct_metrics(
    tmp_path, monkeypatch
):
    import pandas as pd
    import evaluate

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    output_dir = tmp_path / "output"
    train_path = tmp_path / "train.csv"

    # CCO is the generated molecule; CC is included in the training set.
    pd.DataFrame({"smiles": ["CC"]}).to_csv(train_path, index=False)

    manifest = {
        "run_id": "eval-test",
        "strategy": "zero",
        "shots": 0,
        "model": "claude-3-5-haiku-latest",
        "train_path": str(train_path),
        "collected": True,
        "results_file": str(tmp_path / "raw.jsonl"),
        "requests": [
            {
                "custom_id": "req-1",
                "target_logp": 1.0,
                "prompt_example_smiles": [],
            },
            {
                "custom_id": "req-2",
                "target_logp": 2.0,
                "prompt_example_smiles": [],
            },
        ],
    }

    raw_records = [
        {
            "custom_id": "req-1",
            "result_type": "succeeded",
            "text": "CCO",
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "stop_reason": "end_turn",
        },
        {
            "custom_id": "req-2",
            "result_type": "succeeded",
            "text": "not a molecule",
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "stop_reason": "end_turn",
        },
    ]

    (manifest_dir / "eval-test.json").write_text(json.dumps(manifest))
    (tmp_path / "raw.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw_records)
    )

    monkeypatch.setattr(evaluate, "MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(evaluate, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--run-id", "eval-test"]
    )

    evaluate.main()

    detail_path = output_dir / "claude_logp_per_generation.csv"
    summary_path = output_dir / "claude_logp_summary.csv"
    assert detail_path.exists()
    assert summary_path.exists()

    detail = pd.read_csv(detail_path)
    summary = pd.read_csv(summary_path)

    assert len(detail) == 2
    assert set(detail["custom_id"]) == {"req-1", "req-2"}
    assert int(detail["valid"].sum()) == 1

    assert len(summary) == 1
    assert int(summary.loc[0, "n_requests"]) == 2
    assert int(summary.loc[0, "valid_count"]) == 1
    assert summary.loc[0, "validity"] == pytest.approx(0.5)
    assert summary.loc[0, "uniqueness"] == pytest.approx(1.0)
