# Claude Benchmark

A benchmark suite for evaluating Claude models on molecular property-conditioned generation. The initial benchmark focuses on molecular LogP and is designed to expand to properties such as QED, TPSA, and SAS.

## Current benchmark: LogP

The benchmark compares zero-shot generation with random in-context learning (ICL) demonstrations.

| Configuration | Demonstrations | Planned generations |
|---|---:|---:|
| Zero-shot | 0 | 100 |
| Random ICL | 5 | 100 |
| Random ICL | 10 | 100 |
| Random ICL | 100 | 100 |
| Random ICL | 500 | 100 |
| **Total** | | **500** |

These figures describe the prepared experiment design, not confirmed completed runs.

Target LogP values:

- `-1.8906`
- `2.5206`
- `5.1774`
- `7.5889`

ICL demonstrations are sampled randomly from the training data, rather than selected by nearest-neighbor similarity.

## Repository structure

```text
claude-benchmark/
└── scripts/
    └── generation/
        └── baselines/
            └── claude/
                └── logp/
                    ├── common.py
                    ├── submit_batch.py
                    ├── submit_prepared.py
                    ├── collect_batch.py
                    ├── evaluate.py
                    └── langfuse_tracking.py
```

- `common.py`: Shared benchmark configuration and utilities.
- `submit_batch.py`: Batch submission workflow.
- `submit_prepared.py`: Submission of prepared benchmark requests, with an explicit `--submit` safety gate.
- `collect_batch.py`: Collection of completed batch results.
- `evaluate.py`: Evaluation of generated molecules and benchmark metrics.
- `langfuse_tracking.py`: Optional Langfuse telemetry integration.

## Evaluation

The evaluation workflow is designed to measure or report:

- Molecular validity using RDKit.
- Uniqueness and novelty.
- Prompt-copy rate.
- Calculated molecular LogP and absolute error against the target.
- Token usage and available prompt-cache usage statistics.

Use actual evaluation outputs to report benchmark performance. Prepared configurations alone do not establish model performance.

## Setup

Create a Python environment compatible with the dependencies used by the scripts, including the Anthropic SDK, pandas, and RDKit. LiteLLM and Langfuse are used where required by the configured workflow.

From the repository root, inspect the available command-line options:

```bash
python scripts/generation/baselines/claude/logp/submit_prepared.py --help
python scripts/generation/baselines/claude/logp/submit_batch.py --help
python scripts/generation/baselines/claude/logp/collect_batch.py --help
python scripts/generation/baselines/claude/logp/evaluate.py --help
```

Review the available options and request configuration before running a benchmark. Prepared-request submission requires the explicit `--submit` flag.

## Credentials and observability

Configure API credentials through environment variables. Never commit API keys, tokens, `.env` files, or other secrets.

Optional Langfuse configuration:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (optional)
- `LANGFUSE_ENVIRONMENT` (optional)

When the required Langfuse credentials are absent, the tracking helper disables tracking.

## Reproducibility

For each experiment, record the model identifier, target property values, demonstration strategy, number of demonstrations, random seed where applicable, dataset version, and evaluation configuration.

Keep generated results, local datasets, credentials, and virtual environments out of version control unless there is a deliberate reason to publish them.

## Roadmap

- [x] Initial Claude LogP benchmark scripts
- [ ] Document and validate the complete end-to-end workflow
- [ ] Add QED benchmark
- [ ] Add TPSA benchmark
- [ ] Add SAS benchmark
- [ ] Standardize cross-property evaluation and reporting
