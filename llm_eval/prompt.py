"""
prompt.py — what the LLM is shown, and the checks that nothing leaks
=====================================================================

This file decides exactly what the LLM gets to see. Three ideas keep the answers away from it:

1. WHITELIST, NOT BLACKLIST
   `llm_view()` builds a brand-new, minimal copy of each family from problem_dataset.json, field by
   field. Only these fields are copied:
       people:    id, sex, father, mother
       variants:  variant id ("chrom:pos"), gene letter, every person's genotype (null = removed)
   Everything else in the problem file — settings (incl. the random seed that could regenerate the
   original data), the source dataset's name and path, founder genome names, summaries,
   relationship tables — is simply never copied, so it cannot leak.

2. AN INDEPENDENT LEAK CHECK
   `leak_check()` re-reads the answer key (solution_dataset.json) and confirms that, for every cell
   the LLM will be scored on, the view shows null — i.e. the true genotype is not in the prompt.
   It also checks the view only uses whitelisted keys, and that the final prompt text does not
   mention the source dataset or the solution. The app refuses to send a prompt that fails.

3. NO TOOLS
   The LLM call (llm.py) gives the model no tools — no web, no code execution, no files — so the
   prompt below is the only information it has.

The instructions are general: the rules of Mendelian inheritance and what counts as "possible".
They say nothing about any particular family or answer.
"""

import json
from typing import Dict, List, Tuple

from data import variant_id

# ----------------------------------------------------------------------------------------------
# 1) The minimal view of the problem
# ----------------------------------------------------------------------------------------------

# The only keys allowed anywhere in what the LLM receives (checked again in leak_check).
ALLOWED_KEYS = {
    'family', 'people', 'id', 'sex', 'father', 'mother',
    'individuals_with_removed_genotypes', 'variants', 'variant', 'gene', 'genotypes',
}


def family_view(fam: dict) -> dict:
    """One family, reduced to what is needed to solve it (see the whitelist above)."""
    people = [dict(id=p['id'], sex=p.get('sex'), father=p.get('father'), mother=p.get('mother'))
              for p in fam['people']]
    variants = [dict(variant=variant_id(v['chrom'], v['pos']), gene=v['gene'],
                     genotypes={pid: g for pid, g in v['genotypes'].items()})
                for v in fam.get('variants', [])]
    # Who has at least one removed genotype. This is just a convenience: the same information is
    # visible from the nulls themselves.
    removed = sorted({pid for v in variants for pid, g in v['genotypes'].items() if g is None}, key=int)
    return dict(family=fam['family'], people=people, individuals_with_removed_genotypes=removed,
                variants=variants)


def llm_view(problem: dict, families: List[str] = None) -> List[dict]:
    """The view for all families (or only the named ones, when sending one family per request)."""
    return [family_view(f) for f in problem['families']
            if families is None or f['family'] in families]


# ----------------------------------------------------------------------------------------------
# 2) The instructions
# ----------------------------------------------------------------------------------------------
# SYSTEM_PROMPT sets the role. TASK_INSTRUCTIONS explain the data, the rules and the answer format.
# They are fixed text (no dates, names or numbers from the data), so every model and every run gets
# exactly the same instructions — a requirement for a fair, repeatable comparison.

SYSTEM_PROMPT = 'You are an expert in genetics and careful logical reasoning.'

TASK_INSTRUCTIONS = """\
You are given one or more families (pedigrees) and their genotypes at a number of genetic variants.
Some genotypes have been removed. Your task: for every removed genotype, list every genotype that
person could have.

THE DATA
- Each family lists its people with their sex and their father and mother. A parent of null means
  that parent is not part of the family (the person is a founder, or only one parent is known).
- Each variant is one gene with two alleles, written as one letter: the uppercase letter is one
  allele and the lowercase letter the other. With gene "R" the three genotypes are RR, Rr and rr.
- "genotypes" gives each person's genotype at that variant. null means the genotype was removed.
- Variants are named "<chromosome>:<position>", for example "1:446".

THE RULES (Mendelian inheritance)
- Every person carries two alleles. A child receives exactly one allele from its father and exactly
  one from its mother; each parent passes on one of its own two alleles.
- A parent who is not in the family (null) places no constraint on the child.
- Solve each variant on its own. Do not use allele frequencies, population data, or information
  from other variants.
- A genotype is POSSIBLE for a person at a variant if there is at least one way to fill in ALL the
  removed genotypes of that family at that variant such that every parent-child relationship obeys
  the rules above, and in that way the person has that genotype.
- If no consistent way exists at all, the list for that person and variant is empty: [].

WHAT TO ANSWER
- One answer for every person and variant whose genotype is null — and only for those.
- For each, list every possible genotype (each at most once), written with that variant's letter.

OUTPUT FORMAT
Reply with JSON only, exactly in this shape:
{"answers": [
  {"family": "family_001", "variant": "1:446", "individual": "2", "genotypes": ["RR", "Rr"]},
  {"family": "family_001", "variant": "1:446", "individual": "8", "genotypes": ["rr"]}
]}
Use the family names, variant names and individual IDs exactly as given in the data.
"""

# The answer format as a JSON Schema. When the Claude API is used, this is passed as a
# structured-output format so the reply is always valid JSON in exactly this shape. (Answers pasted
# from other LLMs are parsed leniently instead — see scoring.parse_answer.)
ANSWER_SCHEMA = {
    'type': 'object',
    'properties': {
        'answers': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'family': {'type': 'string'},
                    'variant': {'type': 'string'},
                    'individual': {'type': 'string'},
                    'genotypes': {'type': 'array', 'items': {'type': 'string'}},
                },
                'required': ['family', 'variant', 'individual', 'genotypes'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['answers'],
    'additionalProperties': False,
}


def build_user_prompt(view: List[dict]) -> str:
    """Instructions first, then the data as compact JSON (compact = fewer tokens).
    `sort_keys` and fixed separators make the text identical every time for the same problem."""
    data = json.dumps({'families': view}, separators=(',', ':'), sort_keys=True)
    return f'{TASK_INSTRUCTIONS}\nDATA\n{data}\n'


def build_prompt(problem: dict, families: List[str] = None) -> Tuple[str, str, List[dict]]:
    """Returns (system prompt, user prompt, the view used to build it)."""
    view = llm_view(problem, families)
    return SYSTEM_PROMPT, build_user_prompt(view), view


def prompt_as_text(system: str, user: str) -> str:
    """Single text file for pasting into any chat-based LLM by hand."""
    return f'{system}\n\n{user}'


# ----------------------------------------------------------------------------------------------
# 3) The leak check
# ----------------------------------------------------------------------------------------------

def _keys(obj) -> set:
    """Every dictionary key used anywhere inside obj, except the person IDs inside "genotypes"."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            if k != 'genotypes':
                out |= _keys(v)
    elif isinstance(obj, list):
        for x in obj:
            out |= _keys(x)
    return out


def leak_check(view: List[dict], prompt_text: str, problem: dict, solution: dict) -> Dict[str, object]:
    """Run every leak test. Returns {"ok": bool, "checks": [(name, passed, detail), ...]}.

    The solution is read here ONLY to know which cells are scored; nothing from it is added to the
    prompt."""
    checks = []

    # (a) Only whitelisted fields.
    extra = _keys(view) - ALLOWED_KEYS
    checks.append(('Only whitelisted fields are sent', not extra,
                   'extra fields: ' + ', '.join(sorted(extra)) if extra else
                   'people: id, sex, father, mother · variants: id, gene letter, genotypes'))

    # (b) Every scored cell is hidden (null) in the view.
    shown = {(f['family'], v['variant'], pid): g
             for f in view for v in f['variants'] for pid, g in v['genotypes'].items()}
    in_view = {f['family'] for f in view}
    leaked, scored = [], 0
    for fam in solution['families']:
        if fam['family'] not in in_view:
            continue
        for v in fam['variants']:
            vid = variant_id(v['chrom'], v['pos'])
            for pid in v['people']:
                scored += 1
                if shown.get((fam['family'], vid, pid)) is not None:
                    leaked.append(f"{fam['family']} {vid} individual {pid}")
    checks.append(('Every genotype the LLM is scored on is hidden (null)', not leaked,
                   f'{scored:,} scored cells, all null' if not leaked else
                   f'{len(leaked)} visible, e.g. ' + '; '.join(leaked[:5])))

    # (c) The prompt does not name the source dataset, its folder, or the solution.
    src = problem.get('problem', {}) or {}
    words = [w for w in (src.get('source'), src.get('source_path')) if w]
    found = [w for w in words if w in prompt_text]
    for w in ('solution', 'answer_check', 'true_in_possible', 'founder_genome', '"seed"'):
        if w in prompt_text.lower():
            found.append(w)
    checks.append(('No source-dataset name, seed or solution terms in the prompt', not found,
                   'found: ' + ', '.join(found) if found else 'clean'))

    return dict(ok=all(c[1] for c in checks), checks=checks)
