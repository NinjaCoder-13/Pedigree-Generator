"""
data.py — finding and loading problem datasets and their solutions
====================================================================

The evaluation app never runs the simulator or the solver. It only reads two JSON files that the
Pedigree Dataset Generator already writes for every problem:

    outputs/<problem>/problem_dataset.json             the puzzle (removed genotypes are null)
    outputs/<problem>/solution/solution_dataset.json   the answer key (every admissible genotype)

The problem file is the ONLY source for what the LLM is shown (see prompt.py).
The solution file is ONLY used for scoring and for the leak check, and is never sent anywhere.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

# ----------------------------------------------------------------------------------------------
# Where things live
# ----------------------------------------------------------------------------------------------
# APP_DIR      this folder (llm_eval/)
# OUTPUTS      the generator's output folder. Defaults to ../outputs; override with the
#              PED_SIM_OUTPUTS environment variable if your datasets live somewhere else.
APP_DIR = Path(__file__).resolve().parent
OUTPUTS = Path(os.environ.get('PED_SIM_OUTPUTS', APP_DIR.parent / 'outputs')).resolve()

PROBLEM_JSON = 'problem_dataset.json'
SOLUTION_JSON = Path('solution') / 'solution_dataset.json'


# ----------------------------------------------------------------------------------------------
# Listing problems
# ----------------------------------------------------------------------------------------------

def list_problems() -> List[Path]:
    """Every folder under OUTPUTS that contains a problem_dataset.json, newest first."""
    if not OUTPUTS.exists():
        return []
    found = [d for d in OUTPUTS.iterdir() if d.is_dir() and (d / PROBLEM_JSON).exists()]
    return sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)


def has_solution(problem_dir: Path) -> bool:
    """A problem can only be scored after "Solve" was run for it in the generator app."""
    return (problem_dir / SOLUTION_JSON).exists()


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

def load_problem(problem_dir: Path) -> dict:
    """Read problem_dataset.json and make sure it really is a problem (not an original dataset,
    which would contain every genotype and therefore give the answers away)."""
    data = json.loads((problem_dir / PROBLEM_JSON).read_text())
    if data.get('kind') != 'problem':
        raise ValueError(f'{problem_dir.name}/{PROBLEM_JSON} is not a problem dataset '
                         f'(kind = {data.get("kind")!r}). Refusing to use it.')
    return data


def load_solution(problem_dir: Path) -> Optional[dict]:
    """Read solution/solution_dataset.json, or None if the problem has not been solved yet."""
    path = problem_dir / SOLUTION_JSON
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------------------------
# Fingerprints
# ----------------------------------------------------------------------------------------------
# Every saved run records a SHA-256 fingerprint of the problem and solution files it was made and
# scored with. If either file changes later (e.g. the problem is solved again), the app can tell
# that an old score no longer matches the current answer key.

def fingerprint(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def problem_fingerprint(problem_dir: Path) -> Optional[str]:
    return fingerprint(problem_dir / PROBLEM_JSON)


def solution_fingerprint(problem_dir: Path) -> Optional[str]:
    return fingerprint(problem_dir / SOLUTION_JSON)


def variant_id(chrom, pos) -> str:
    """How a variant is named everywhere in this app (prompt, answers, scoring): "<chrom>:<pos>"."""
    return f'{chrom}:{int(pos)}'
