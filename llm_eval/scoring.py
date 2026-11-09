"""
scoring.py — turning an LLM's answer into hypotheses and scoring them
======================================================================

Everything here is deterministic: the same answer + the same answer key always gives exactly the
same scores. Nothing is random, nothing depends on order, and no LLM is involved in grading.

KEY WORDS
---------
cell        one (family, variant, individual) whose genotype was removed and is scored.
hypothesis  one genotype the LLM proposes for one cell, e.g. "individual 2 at 1:446 could be RR".
            An answer listing ["RR", "Rr"] for a cell is TWO hypotheses.
admissible  a hypothesis that is in the answer key (solution_dataset.json): the genotype really
            is possible for that cell under Mendelian inheritance.
admissible space
            the set of ALL admissible hypotheses = for every cell, every possible genotype.
            Example: cell A can be RR or Rr, cell B can only be rr  ->  space size 3.

THE THREE RATES
---------------
Let N = every hypothesis the LLM suggested, counting repeats.

Validity rate   = (suggested hypotheses that are admissible, counting repeats) / N  x 100
                  "Of everything it said, how much was right?"
Uniqueness rate = (distinct hypotheses) / N  x 100
                  "Does it keep repeating itself?" 100% = no repeats.
Recovery rate   = (distinct admissible hypotheses found) / (size of admissible space)  x 100
                  "How much of the full answer did it find?"

A hypothesis is INVALID when (reason in brackets):
  [impossible]    the cell is real but that genotype is not possible there,
  [not_a_target]  the family/variant/individual does not exist, or that genotype was not removed
                  (the LLM answered a question that wasn't asked),
  [malformed]     the genotype is not two letters of that variant's gene (e.g. "AB", "R", "Xx" for gene R).
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data import variant_id

# ----------------------------------------------------------------------------------------------
# 1) The answer key
# ----------------------------------------------------------------------------------------------

def admissible_space(solution: dict) -> Tuple[Dict[tuple, set], Dict[tuple, str]]:
    """From solution_dataset.json build
         space[(family, variant, individual)] = {"RR", "Rr"}   (empty set = no consistent genotype)
         genes[(family, variant)] = "R"                         (the gene letter, for normalising)"""
    space, genes = {}, {}
    for fam in solution['families']:
        for v in fam['variants']:
            vid = variant_id(v['chrom'], v['pos'])
            genes[(fam['family'], vid)] = v['gene']
            for pid, cell in v['people'].items():
                space[(fam['family'], vid, str(pid))] = set(cell['possible'])
    return space, genes


# ----------------------------------------------------------------------------------------------
# 2) Reading the LLM's reply
# ----------------------------------------------------------------------------------------------

def _extract_json(text: str):
    """Find the JSON in a reply. Handles a bare JSON reply, ```json fenced``` blocks, and JSON
    surrounded by explanation text (takes the outermost {...} or [...])."""
    text = text.strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    candidates = [text] + ([fence.group(1)] if fence else [])
    for open_, close in (('{', '}'), ('[', ']')):
        i, j = text.find(open_), text.rfind(close)
        if 0 <= i < j:
            candidates.append(text[i:j + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError('No readable JSON found in the reply.')


def _genotype_list(gts) -> Optional[List[str]]:
    """A genotype list as written by the LLM. Tolerates "RR/Rr" or "RR, Rr" written as one string."""
    if isinstance(gts, str):
        gts = [g for g in re.split(r'[/,;\s]+', gts) if g]
    return [str(g) for g in gts] if isinstance(gts, list) else None


def parse_answer(text: str) -> Tuple[List[dict], Optional[str]]:
    """Reply text -> list of answer entries {family, variant, individual, genotypes}, one per person
    the LLM answered for (so both answer formats are scored in exactly the same way).

    Returns (entries, problem) where `problem` is a short note if the reply could not be read fully.
    Accepted shapes (an "answers" list, or the list on its own), with entries in either format:
      {family, variant, groups: [{genotypes, individuals}, ...]}   one per variant (current format)
      {family, variant, individual, genotypes}                     one per person (prompt version 1)
    A person listed in two groups of the same variant gets both genotype lists (any genotype they
    share then counts as a repeat). An unreadable reply gives zero entries = zero hypotheses."""
    try:
        data = _extract_json(text)
    except ValueError as e:
        return [], str(e)
    items = data.get('answers') if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], 'The JSON has no "answers" list.'
    entries, skipped = [], 0
    for it in items:
        if not isinstance(it, dict):
            skipped += 1
            continue
        fam, vid = str(it.get('family', '')), str(it.get('variant', ''))
        if 'groups' in it:
            groups = it['groups'] if isinstance(it['groups'], list) else []
            if not groups and it['groups'] != []:
                skipped += 1
            for g in groups:
                gts = _genotype_list(g.get('genotypes', [])) if isinstance(g, dict) else None
                people = g.get('individuals') if isinstance(g, dict) else None
                if isinstance(people, (str, int)):
                    people = [people]
                if gts is None or not isinstance(people, list):
                    skipped += 1
                    continue
                entries += [dict(family=fam, variant=vid, individual=str(pid), genotypes=list(gts))
                            for pid in people]
        else:
            gts = _genotype_list(it.get('genotypes', []))
            if gts is None:
                skipped += 1
                continue
            entries.append(dict(family=fam, variant=vid, individual=str(it.get('individual', '')),
                                genotypes=gts))
    note = f'{skipped} answer entries were not in the expected shape and were ignored.' if skipped else None
    return entries, note


# ----------------------------------------------------------------------------------------------
# 3) Normalising one genotype
# ----------------------------------------------------------------------------------------------

def normalise_genotype(raw: str, gene: Optional[str]) -> Optional[str]:
    """'rR' -> 'Rr', 'r/R' -> 'Rr', ' rr ' -> 'rr'. Returns None if it is not two letters of `gene`.
    Uppercase is always written first, so 'Rr' and 'rR' are the same hypothesis."""
    if gene is None:
        return None
    letters = re.sub(r'[\s/|,]', '', raw)
    if len(letters) != 2 or any(ch.lower() != gene.lower() for ch in letters):
        return None
    n_upper = sum(ch.isupper() for ch in letters)
    return {2: gene.upper() * 2, 1: gene.upper() + gene.lower(), 0: gene.lower() * 2}[n_upper]


# ----------------------------------------------------------------------------------------------
# 4) Scoring
# ----------------------------------------------------------------------------------------------

def hypotheses_table(entries: List[dict], space: Dict[tuple, set], genes: Dict[tuple, str]) -> pd.DataFrame:
    """One row per suggested hypothesis, in the order the LLM gave them, with its verdict."""
    rows = []
    for k, e in enumerate(entries):
        fam, vid, pid = e['family'].strip(), e['variant'].strip(), e['individual'].strip()
        gene = genes.get((fam, vid))
        cell = (fam, vid, pid)
        for raw in e['genotypes']:
            g = normalise_genotype(raw, gene)
            if g is None:
                verdict = 'malformed' if gene is not None else 'not_a_target'
            elif cell not in space:
                verdict = 'not_a_target'
            elif g in space[cell]:
                verdict = 'valid'
            else:
                verdict = 'impossible'
            rows.append(dict(entry=k, family=fam, variant=vid, individual=pid, genotype_raw=raw,
                             genotype=g if g is not None else raw.strip(), verdict=verdict,
                             valid=verdict == 'valid'))
    df = pd.DataFrame(rows, columns=['entry', 'family', 'variant', 'individual', 'genotype_raw',
                                     'genotype', 'verdict', 'valid'])
    # A hypothesis is a repeat if exactly the same (cell, genotype) appeared earlier in the reply.
    df['repeat'] = df.duplicated(subset=['family', 'variant', 'individual', 'genotype'], keep='first')
    return df


def _pct(a: int, b: int) -> Optional[float]:
    return round(100.0 * a / b, 2) if b else None


def score(entries: List[dict], solution: dict) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The main entry point. Returns
         metrics  the three rates + every count behind them (overall)
         hyps     one row per suggested hypothesis with its verdict (see hypotheses_table)
         per_fam  the same metrics per family"""
    space, genes = admissible_space(solution)
    hyps = hypotheses_table(entries, space, genes)
    answered = answered_cells(entries, space)
    metrics = _metrics(hyps, space, answered)

    per_fam = []
    for fam in sorted({k[0] for k in space}):
        sub_space = {k: v for k, v in space.items() if k[0] == fam}
        m = _metrics(hyps[hyps['family'] == fam], sub_space, answered)
        per_fam.append(dict(family=fam, **{k: m[k] for k in (
            'variants_complete', 'variants_partial', 'variants_missing', 'variants_total',
            'cells_answered', 'cells', 'suggested', 'valid', 'distinct', 'distinct_valid',
            'admissible_space', 'validity_rate', 'uniqueness_rate', 'recovery_rate')}))
    return metrics, hyps, pd.DataFrame(per_fam)


# ----------------------------------------------------------------------------------------------
# 5) Coverage: which removed genotypes / variants the LLM answered at all
# ----------------------------------------------------------------------------------------------
# A cell counts as ANSWERED when the reply lists that person at that variant, whatever genotypes
# it gives (even [] or wrong ones). A variant is
#   complete   every removed genotype of it was answered
#   partial    some were answered
#   missing    none were answered (the LLM never got to it, or skipped it)
# Coverage says nothing about correctness; that is what the three rates are for.

def answered_cells(entries: List[dict], space: Dict[tuple, set]) -> set:
    return {(e['family'].strip(), e['variant'].strip(), e['individual'].strip()) for e in entries} & set(space)


def variant_coverage(entries: List[dict], solution: dict) -> pd.DataFrame:
    """One row per (family, variant) with removed genotypes: how many were answered, and the status."""
    space, _ = admissible_space(solution)
    return _coverage_table(space, answered_cells(entries, space))


def _coverage_table(space: Dict[tuple, set], answered: set) -> pd.DataFrame:
    cells, done = Counter(), Counter()
    for f, v, p in space:
        cells[(f, v)] += 1
        done[(f, v)] += (f, v, p) in answered
    rows = [dict(family=f, variant=v, cells=cells[(f, v)], answered=done[(f, v)],
                 status='complete' if done[(f, v)] == cells[(f, v)] else 'partial' if done[(f, v)] else 'missing')
            for f, v in cells]                                  # dict order = answer-key order
    return pd.DataFrame(rows, columns=['family', 'variant', 'cells', 'answered', 'status'])


_KEY = re.compile(r'"(family|variant)"\s*:\s*"([^"]*)"')


def variants_in_text(text: str, default_family: str = '') -> set:
    """Live coverage while a reply is still streaming in: every (family, variant) the reply has
    started to answer so far. Works on unfinished JSON (just looks for "family": "…" and
    "variant": "…" in order; each variant belongs to the family named last before it)."""
    found, fam = set(), default_family
    for key, value in _KEY.findall(text):
        if key == 'family':
            fam = value
        else:
            found.add((fam, value))
    return found


def _metrics(hyps: pd.DataFrame, space: Dict[tuple, set], answered: set) -> dict:
    n = len(hyps)                                           # every suggestion, repeats included
    n_valid = int(hyps['valid'].sum())                       # admissible suggestions, repeats included
    distinct = hyps[~hyps['repeat']]
    n_distinct = len(distinct)                               # different suggestions
    n_distinct_valid = int(distinct['valid'].sum())          # different admissible suggestions found
    space_size = sum(len(v) for v in space.values())         # every admissible hypothesis
    answered = answered & set(space)
    status = Counter(_coverage_table(space, answered)['status'])
    verdicts = Counter(hyps['verdict'])
    return dict(
        validity_rate=_pct(n_valid, n),
        uniqueness_rate=_pct(n_distinct, n),
        recovery_rate=_pct(n_distinct_valid, space_size),
        suggested=n, valid=n_valid, invalid=n - n_valid,
        distinct=n_distinct, repeats=n - n_distinct, distinct_valid=n_distinct_valid,
        admissible_space=space_size, missed=space_size - n_distinct_valid,
        invalid_impossible=verdicts.get('impossible', 0),
        invalid_not_a_target=verdicts.get('not_a_target', 0),
        invalid_malformed=verdicts.get('malformed', 0),
        cells=len(space), cells_answered=len(answered),
        conflict_cells=sum(1 for v in space.values() if not v),
        variants_total=sum(status.values()), variants_complete=status.get('complete', 0),
        variants_partial=status.get('partial', 0), variants_missing=status.get('missing', 0),
    )


def missed_hypotheses(hyps: pd.DataFrame, solution: dict) -> pd.DataFrame:
    """Every admissible hypothesis the LLM did NOT find (what is missing from its recovery)."""
    space, _ = admissible_space(solution)
    found = set(map(tuple, hyps.loc[hyps['valid'], ['family', 'variant', 'individual', 'genotype']].values))
    rows = [dict(family=f, variant=v, individual=p, genotype=g)
            for (f, v, p), gs in sorted(space.items()) for g in sorted(gs) if (f, v, p, g) not in found]
    return pd.DataFrame(rows, columns=['family', 'variant', 'individual', 'genotype'])
