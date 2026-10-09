import os
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen


# ============================================================
# PATHS
# ============================================================

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("CFG_TEST_ROOT", HERE.parents[4])
)

DATA_ROOT = Path(
    os.environ.get(
        "GUACAMOL_500K_DIR",
        REPO_ROOT / "data" / "guacamol_500k",
    )
)

TRAIN_PATH = DATA_ROOT / "train_500k.csv"
TEST_PATH = DATA_ROOT / "test_500k.csv"

OUTPUT_DIR = Path(
    os.environ.get("CLAUDE_OUTPUT_DIR", HERE / "results")
)

MANIFEST_DIR = OUTPUT_DIR / "manifests"


# ============================================================
# EXPERIMENT
# ============================================================

MODEL = os.environ.get(
    "CLAUDE_MODEL",
    "claude-3-5-haiku-latest",
)

TARGETS = [
    -1.8906,
    2.5206,
    5.1774,
    7.5889,
]

ALLOWED_SHOTS = [0, 5, 10, 100, 500]

N_PER_TARGET = 25
SEED = 42
MAX_TOKENS = 256
TEMPERATURE = 0.7


# ============================================================
# RDKit HELPERS
# ============================================================

def canonicalize(smiles):
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True,
        )

    except Exception:
        return None


def calc_logp(smiles):
    canonical = canonicalize(smiles)

    if canonical is None:
        return None

    mol = Chem.MolFromSmiles(canonical)

    if mol is None:
        return None

    return float(Crippen.MolLogP(mol))


# ============================================================
# SMILES EXTRACTION
# ============================================================

def extract_smiles(text):
    if not isinstance(text, str):
        return None

    text = text.strip()

    if not text:
        return None

    fenced = re.findall(
        r"```(?:smiles)?\s*([^`]+?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    candidates = fenced + text.splitlines() + [text]

    for candidate in candidates:
        line = candidate.strip()

        if not line:
            continue

        line = re.sub(
            r"^(SMILES\s*[:=]\s*)",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()

        line = line.strip("`'\"").strip()

        if len(line) < 2:
            continue

        canonical = canonicalize(line)

        if canonical is not None:
            return canonical

    return None


# ============================================================
# DATASET
# ============================================================

def load_train():
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    required = {"smiles"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected a smiles column in {TRAIN_PATH}"
        )

    df = df.dropna(subset=["smiles"]).copy()

    df["smiles"] = df["smiles"].astype(str).str.strip()

    df = df[df["smiles"].str.len() > 0].copy()

    if "logp" in df.columns:
        df["logp"] = pd.to_numeric(
            df["logp"],
            errors="coerce",
        )
    else:
        print("LogP column missing. Computing LogP with RDKit.")
        df["logp"] = df["smiles"].map(calc_logp)

    df = df.dropna(subset=["logp"])

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable training molecules found.")

    return df[["smiles", "logp"]]


def build_example_orders(train_df, seed=SEED):
    """
    Random examples are deterministic and nested across shot counts.
    Nearest examples are ranked by absolute LogP distance.
    """

    orders = {
        "random": {},
        "nearest": {},
    }

    for target_idx, target in enumerate(TARGETS):

        random_order = train_df.sample(
            frac=1.0,
            random_state=seed + target_idx,
        ).reset_index(drop=True)

        nearest_order = (
            train_df.assign(
                _distance=(
                    train_df["logp"] - target
                ).abs()
            )
            .sort_values(
                "_distance",
                kind="mergesort",
            )
            .drop(columns="_distance")
            .reset_index(drop=True)
        )

        orders["random"][target_idx] = random_order
        orders["nearest"][target_idx] = nearest_order

    return orders


def select_examples(
    orders,
    strategy,
    target_idx,
    shots,
):
    if strategy == "zero":
        if shots != 0:
            raise ValueError(
                "Zero-shot requires --shots 0."
            )

        return pd.DataFrame(
            columns=["smiles", "logp"]
        )

    if strategy not in ("random", "nearest"):
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    if shots not in ALLOWED_SHOTS or shots == 0:
        raise ValueError(
            f"Unsupported shot count: {shots}"
        )

    examples = orders[strategy][target_idx]

    if shots > len(examples):
        raise ValueError(
            f"Only {len(examples)} training examples available."
        )

    return examples.head(shots).reset_index(drop=True)


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a molecular generation model.

Generate exactly ONE valid molecular SMILES string whose
calculated octanol/water partition coefficient (LogP), calculated
using RDKit Crippen MolLogP, is as close as possible to the
requested target.

A valid SMILES is not sufficient by itself: try to match the
requested LogP while producing a chemically plausible molecule.

Treat the examples as demonstrations of the requested input-output
format. For each example, the supplied LogP is the reference
property value for that example.

Do not copy an example unnecessarily.

Return only the generated SMILES. Do not include explanations,
Markdown fences, labels, or multiple candidates.
""".strip()


def build_user_content(
    target_logp,
    examples,
    shots,
):
    blocks = []

    if shots > 0:

        lines = [
            "Training examples:"
        ]

        for i, row in enumerate(
            examples.to_dict("records"),
            start=1,
        ):
            lines.append(
                f"Example {i}\n"
                f"LogP: {float(row['logp']):.4f}\n"
                f"SMILES: {row['smiles']}"
            )

        example_text = "\n\n".join(lines)

        example_block = {
            "type": "text",
            "text": example_text,
        }

        # Cache longer demonstration prefixes.
        # Smaller prompts are left uncached because they may be
        # below the model's minimum cacheable prefix length.
        if shots >= 100:
            example_block["cache_control"] = {
                "type": "ephemeral",
                "ttl": "1h",
            }

        blocks.append(example_block)

    blocks.append({
        "type": "text",
        "text": (
            "Generate exactly one molecule for the following target.\n\n"
            f"Target LogP: {float(target_logp):.4f}\n\n"
            "Return only the generated SMILES."
        ),
    })

    return blocks


# ============================================================
# EVALUATION HELPERS
# ============================================================

def canonical_training_set(train_df):
    canonical = set()

    for smiles in train_df["smiles"]:
        value = canonicalize(smiles)

        if value is not None:
            canonical.add(value)

    return canonical


def safe_name(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )
