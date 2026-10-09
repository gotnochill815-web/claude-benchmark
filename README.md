# Claude Molecular Property ICL Benchmark

A benchmark for evaluating Claude models on molecular
property-conditioned generation. The initial experiment targets RDKit
Crippen LogP. QED, TPSA, and SAS are future extensions, not currently
implemented benchmarks.

## Current experiment: LogP

The benchmark compares zero-shot generation against random in-context
learning (ICL) demonstrations.

  Experiment     Demonstrations   Planned generations
  ------------ ---------------- ---------------------
  Zero-shot                   0                   100
  Random ICL                  5                   100
  Random ICL                 10                   100
  Random ICL                100                   100
  Random ICL                500                   100
  **Total**                                   **500**

Each configuration uses four target LogP values and 25 generations per
target. These numbers describe the experiment design, not completed
model outputs.

Targets:

-   `-1.8906`
-   `2.5206`
-   `5.1774`
-   `7.5889`

ICL examples are selected randomly from the training dataset. They are
not selected by nearest-neighbor similarity. Random selection is seeded
for reproducibility.

## Repository structure

``` text
scripts/generation/baselines/claude/logp/
├── common.py
├── zero_shot.py
├── random_5shot.py
├── random_10shot.py
├── random_100shot.py
├── random_500shot.py
├── submit_batch.py
├── submit_prepared.py
├── collect_batch.py
├── evaluate.py
└── langfuse_tracking.py
```

-   `common.py`: Paths, configuration, dataset loading, SMILES
    processing, prompt construction, and shared helpers.
-   `zero_shot.py`: Zero-shot experiment.
-   `random_*shot.py`: Random ICL experiments with fixed shot counts.
-   `submit_batch.py`: Shared request preparation and optional batch
    submission.
-   `submit_prepared.py`: Submit a previously prepared manifest.
-   `collect_batch.py`: Collect results from completed Anthropic
    batches.
-   `evaluate.py`: Calculate molecular validity, uniqueness, novelty,
    prompt-copy rate, LogP error, and token/cache usage summaries.
-   `langfuse_tracking.py`: Optional Langfuse telemetry.

## Requirements

Use a Python environment with:

-   Anthropic Python SDK
-   pandas
-   RDKit

Langfuse is optional and is used only when configured. Install
dependencies according to the environment where the benchmark will run.

## Dataset and configuration

By default, the scripts expect:

-   Training data: `data/guacamol_500k/train_500k.csv`
-   Test data: `data/guacamol_500k/test_500k.csv`

The CSV files must contain a `smiles` column. A `logp` column is used
when available; otherwise, LogP is calculated with RDKit.

Configuration can be overridden using:

-   `CFG_TEST_ROOT`: repository root
-   `GUACAMOL_500K_DIR`: directory containing the dataset CSV files
-   `CLAUDE_OUTPUT_DIR`: directory for manifests, request JSONL, and
    results
-   `CLAUDE_MODEL`: default model identifier

The default model is defined in `common.py`. Explicitly set
`CLAUDE_MODEL` when you need to match an existing run's model.

## Workflow

Run commands from the repository root.

### 1. Prepare a configuration (dry run)

The following commands prepare requests and manifests. They do not
submit API batches unless `--submit` is supplied.

``` bash
python scripts/generation/baselines/claude/logp/zero_shot.py
python scripts/generation/baselines/claude/logp/random_5shot.py
python scripts/generation/baselines/claude/logp/random_10shot.py
python scripts/generation/baselines/claude/logp/random_100shot.py
python scripts/generation/baselines/claude/logp/random_500shot.py
```

Inspect the printed run ID, model, request count, JSONL path, and
manifest path. Review the manifest and request payloads before
submitting anything.

### 2. Submit a prepared run

Set `ANTHROPIC_API_KEY` securely in the environment. Never commit
credentials.

``` bash
python scripts/generation/baselines/claude/logp/submit_prepared.py \
  --run-id YOUR_RUN_ID --submit
```

Submission incurs API usage and potential charges. Do not add `--submit`
until you have checked the model, shot count, and request payloads.

### 3. Collect completed results

``` bash
python scripts/generation/baselines/claude/logp/collect_batch.py \
  --run-id YOUR_RUN_ID
```

Omit `--run-id` to check all manifests for submitted batches.

### 4. Evaluate collected results

``` bash
python scripts/generation/baselines/claude/logp/evaluate.py \
  --run-id YOUR_RUN_ID
```

Omit `--run-id` to evaluate all collected runs. Outputs include
per-generation details and a summary CSV in the configured output
directory.

## Metrics

-   **Validity:** fraction of requests that yield a parseable molecular
    SMILES.
-   **Uniqueness:** fraction of valid outputs that are unique after
    canonicalization.
-   **Novelty:** fraction of valid outputs absent from the canonical
    training set.
-   **Prompt-copy rate:** fraction of valid outputs matching a
    demonstration SMILES.
-   **Mean absolute LogP error:** average absolute difference between
    calculated output LogP and the requested target.
-   **Usage:** input/output tokens and available cache read/creation
    token counts.

Novelty is measured against the training set. The test CSV is configured
for the benchmark but is not currently used to calculate novelty.

## Optional Langfuse tracking

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable tracking.
Optional settings:

-   `LANGFUSE_HOST`
-   `LANGFUSE_ENVIRONMENT`

Without the required credentials, tracking is disabled. Telemetry is for
benchmark observations and usage metadata, not hidden chain-of-thought.

## Reproducibility and safety

Record the model identifier, dataset version, target values, shot count,
selection strategy, seed, and evaluation configuration for every
comparison. Keep local datasets, generated results, virtual
environments, API keys, and `.env` files out of version control.

Prepared configurations are not evidence of completed runs or model
performance.

## Roadmap

-   [x] Initial Claude LogP benchmark scripts
-   [ ] Validate the complete end-to-end workflow
-   [ ] Add QED benchmark
-   [ ] Add TPSA benchmark
-   [ ] Add SAS benchmark
-   [ ] Standardize cross-property evaluation and reporting
