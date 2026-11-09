"""
runs.py — saving and re-loading evaluation runs
===============================================

Every evaluation is saved as a folder so it can be inspected, re-scored and compared later:

    llm_eval/runs/<problem name>/<YYYYmmdd_HHMMSS>_<label>/
        run.json            who/what/when: model, effort, how the answer was obtained, fingerprints
                            of the problem and answer key, token usage
        prompt.txt          the exact text sent (system + user prompt) — for one-request runs
        prompt_<part>.txt   the exact text sent per request — when the problem was split, <part> is
                            the family (one request per family) or <family>__<chrom>-<pos> (one
                            request per variant)
        reply.txt / reply_<part>.txt
                            the exact reply text(s) from the LLM, unchanged
        metrics.json        the three rates and every count behind them
        hypotheses.csv      one row per suggested hypothesis with its verdict
        per_family.csv      the rates (and variant coverage) per family
        coverage.csv        one row per variant: how many of its removed genotypes were answered
        missed.csv          admissible hypotheses the LLM did not find

WHY OUTSIDE outputs/<problem>/ ?  The generator app zips a problem for handing out ("Problem only").
Graded runs contain verdicts, i.e. parts of the answer key, so they are kept in a separate folder
that is never included in that ZIP. The folder is also git-ignored.

The reply files are the source of truth: `rescore()` reads them again and recomputes every score,
which is how determinism can be checked at any time.
"""

import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

import data as D
import scoring as S

RUNS = D.APP_DIR / 'runs'


def run_root(problem_dir: Path) -> Path:
    return RUNS / problem_dir.name


def list_runs(problem_dir: Path) -> List[Path]:
    root = run_root(problem_dir)
    if not root.exists():
        return []
    return sorted((d for d in root.iterdir() if (d / 'run.json').exists()), reverse=True)


def new_run_dir(problem_dir: Path, label: str) -> Path:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', label.strip()).strip('_') or 'run'
    d = run_root(problem_dir) / f'{dt.datetime.now():%Y%m%d_%H%M%S}_{safe}'
    d.mkdir(parents=True)
    return d


def save_run(problem_dir: Path, label: str, info: dict, prompts: Dict[str, str],
             replies: Dict[str, str]) -> Path:
    """Write a new run and score it.
       prompts / replies are keyed by part: '' for a single request, or the family name when each
       family was sent separately."""
    d = new_run_dir(problem_dir, label)
    for part, text in prompts.items():
        (d / (f'prompt_{part}.txt' if part else 'prompt.txt')).write_text(text)
    for part, text in replies.items():
        (d / (f'reply_{part}.txt' if part else 'reply.txt')).write_text(text)
    info = dict(info, label=label, problem=problem_dir.name, parts=sorted(replies),
                created=dt.datetime.now().isoformat(timespec='seconds'),
                problem_fingerprint=D.problem_fingerprint(problem_dir))
    (d / 'run.json').write_text(json.dumps(info, indent=2))
    rescore(problem_dir, d)
    return d


def read_replies(run_dir: Path) -> Dict[str, str]:
    info = json.loads((run_dir / 'run.json').read_text())
    return {p: (run_dir / (f'reply_{p}.txt' if p else 'reply.txt')).read_text() for p in info['parts']}


def rescore(problem_dir: Path, run_dir: Path) -> dict:
    """(Re)compute every score from the saved reply text(s) and the current answer key."""
    solution = D.load_solution(problem_dir)
    if solution is None:
        raise ValueError('This problem has no solution yet. Run "Solve" in the generator app first.')
    entries, notes = [], []
    for part, text in read_replies(run_dir).items():
        got, note = S.parse_answer(text)
        entries += got
        if note:
            notes.append(f'{part or "reply"}: {note}')
    metrics, hyps, per_fam = S.score(entries, solution)
    metrics['parse_notes'] = notes
    metrics['answer_entries'] = len(entries)
    metrics['solution_fingerprint'] = D.solution_fingerprint(problem_dir)
    metrics['scored'] = dt.datetime.now().isoformat(timespec='seconds')
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    hyps.to_csv(run_dir / 'hypotheses.csv', index=False)
    S.variant_coverage(entries, solution).to_csv(run_dir / 'coverage.csv', index=False)
    per_fam.to_csv(run_dir / 'per_family.csv', index=False)
    S.missed_hypotheses(hyps, solution).to_csv(run_dir / 'missed.csv', index=False)
    return metrics


def load_run(run_dir: Path) -> dict:
    """Everything about a saved run, for display."""
    out = dict(dir=run_dir, info=json.loads((run_dir / 'run.json').read_text()))
    out['metrics'] = json.loads((run_dir / 'metrics.json').read_text()) if (run_dir / 'metrics.json').exists() else {}
    for name in ('hypotheses', 'per_family', 'missed', 'coverage'):
        p = run_dir / f'{name}.csv'
        out[name] = pd.read_csv(p, dtype={'individual': str, 'variant': str}) if p.exists() else pd.DataFrame()
    return out


def leaderboard(problem_dir: Path) -> pd.DataFrame:
    """One row per saved run of this problem, for comparing models side by side."""
    rows = []
    for d in list_runs(problem_dir):
        r = load_run(d)
        i, m = r['info'], r['metrics']
        rows.append(dict(run=d.name, label=i.get('label'), source=i.get('source'),
                         model=i.get('served_by') or i.get('model'), effort=i.get('effort'),
                         temperature=i.get('temperature'), seed=i.get('seed'),
                         mode=i.get('mode'), prompt_version=i.get('prompt_version', 1),
                         variants_covered=coverage_text(m),
                         variants_complete=m.get('variants_complete'), variants_total=m.get('variants_total'),
                         cells_answered=m.get('cells_answered'), cells=m.get('cells'),
                         validity_rate=m.get('validity_rate'), uniqueness_rate=m.get('uniqueness_rate'),
                         recovery_rate=m.get('recovery_rate'), suggested=m.get('suggested'),
                         admissible_space=m.get('admissible_space'),
                         failed_requests=len(i.get('failed_parts') or {}),
                         run_seconds=i.get('wall_seconds', i.get('seconds')), request_seconds=i.get('seconds'),
                         cost_usd=i.get('cost_usd'), cost_complete=i.get('cost_complete'),
                         output_tokens=i.get('output_tokens'),
                         up_to_date=m.get('solution_fingerprint') == D.solution_fingerprint(problem_dir)))
    return pd.DataFrame(rows)


def coverage_text(m: dict) -> str:
    """E.g. "57 / 148 (+2 partial)" — variants fully answered out of all variants to answer."""
    if m.get('variants_total') is None:
        return '— (re-score)'
    extra = f" (+{m['variants_partial']} partial)" if m.get('variants_partial') else ''
    return f"{m['variants_complete']} / {m['variants_total']}{extra}"
