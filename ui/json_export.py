"""
One JSON file per generated dataset, problem dataset and solution, holding every family.

    <dataset>/dataset.json                     generated dataset: people, pedigree, relationships and
                                               every person's genotype at every variant
    <problem>/problem_dataset.json             the same for a problem dataset (removed genotypes = null)
    <problem>/solution/solution_dataset.json   every possible genotype of every removed person, per variant

The JSON files are built from the other files in the folder, so they can be (re)made at any time:

    python ui/json_export.py outputs/<folder> [more folders...]
    python ui/json_export.py --all
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

DATASET_JSON = 'dataset.json'
PROBLEM_JSON = 'problem_dataset.json'
SOLUTION_JSON = 'solution_dataset.json'
FORMAT_VERSION = 1


def _num(x):
    """Plain int/float for JSON (pandas/numpy scalars and 'NA' become int/float/None)."""
    if x is None or (isinstance(x, float) and x != x):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None if str(x) in ('', 'NA', 'nan') else str(x)
    return int(f) if f.is_integer() and abs(f) < 2 ** 53 else f


def _genotype(cell) -> Optional[str]:
    return None if not isinstance(cell, str) or cell in ('', 'NA', './.') else cell


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def _family_sort_key(name: str):
    return [int(t) if t.isdigit() else t for t in name.replace('_', ' ').split()]


def family_dirs(run_dir: Path) -> List[Path]:
    return sorted((d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('family_')),
                  key=lambda d: _family_sort_key(d.name))


def family_record(fam_dir: Path, masked: Optional[List[str]] = None) -> dict:
    """Everything about one family: people, parents, relationships and genotypes per variant."""
    import pipeline as P  # local import: pipeline imports this module too

    G = nx.read_edgelist(str(fam_dir / 'pedigree.nx'), create_using=nx.DiGraph())
    prof = pd.read_csv(fam_dir / 'profiles.txt', sep='\t', dtype={'ID': str})
    for n in prof['ID']:
        G.add_node(n)
    sex = dict(zip(prof['ID'], prof['Sex']))
    year = dict(zip(prof['ID'], prof['Gen']))
    layer = P.generation_layers(G, prof)
    masked = set(map(str, masked or []))

    founder_vcf = {}
    if (fam_dir / 'founder_genomes.txt').exists():
        fg = _read_tsv(fam_dir / 'founder_genomes.txt')
        founder_vcf = dict(zip(fg.iloc[:, 0], fg.iloc[:, 1]))

    people = []
    for n in sorted(G.nodes, key=int):
        ps = sorted(G.predecessors(n), key=int)
        father = next((p for p in ps if sex.get(p) == 'male'), None)
        mother = next((p for p in ps if p != father), None)
        person = dict(id=n, sex=sex.get(n), birth_year=_num(year.get(n)), generation=layer.get(n),
                      father=father, mother=mother, founder=not ps,
                      children=sorted(G.successors(n), key=int))
        if n in founder_vcf:
            person['founder_genome'] = founder_vcf[n]
        if masked:
            person['genotype_removed'] = n in masked
        people.append(person)

    rec = dict(family=fam_dir.name, n_people=len(people), people=people)

    gt_path = fam_dir / 'genotypes_letters.tsv.gz'
    if gt_path.exists():
        gt = _read_tsv(gt_path)
        ids = [p['id'] for p in people if p['id'] in gt.columns]
        rec['n_variants'] = len(gt)
        rec['variants'] = [
            dict(chrom=str(row['CHROM']), pos=int(row['POS']), gene=row['GENE'],
                 genotypes={i: _genotype(row[i]) for i in ids})
            for _, row in gt.iterrows()]

    rel_path = fam_dir / 'relationships.csv'
    if rel_path.exists():
        rel = pd.read_csv(rel_path, dtype={'ID1': str, 'ID2': str})
        rec['relationships'] = [
            dict(id1=r.ID1, id2=r.ID2, relationship=r.Relationship, meioses=_num(r.MC),
                 generation_gap=_num(r.GD), type=r.RT)
            for r in rel.itertuples(index=False)]
    return rec


def _summary(run_dir: Path) -> List[dict]:
    path = run_dir / 'summary.csv'
    if not path.exists():
        return []
    return [{k: (_num(v) if k != 'family' else v) for k, v in row.items()}
            for row in pd.read_csv(path).to_dict('records')]


def write_dataset_json(run_dir: Path) -> Path:
    """dataset.json for a generated dataset, or problem_dataset.json for a problem dataset."""
    run_dir = Path(run_dir)
    problem_path = run_dir / 'problem.json'
    problem = json.loads(problem_path.read_text()) if problem_path.exists() else None
    settings = json.loads((run_dir / 'settings.json').read_text())
    settings.pop('problem', None)
    masked = (problem or {}).get('masked', {})
    out = dict(
        format_version=FORMAT_VERSION,
        kind='problem' if problem else 'dataset',
        name=run_dir.name,
        genotype_codes={'AA': 'both copies dominant (1|1)', 'Aa': 'one of each (1|0 or 0|1)',
                        'aa': 'both copies recessive (0|0)', 'null': 'genotype removed / unknown'},
        settings=settings,
        summary=_summary(run_dir),
    )
    if problem:
        out['problem'] = problem
    out['families'] = [family_record(d, masked.get(d.name)) for d in family_dirs(run_dir)]
    path = run_dir / (PROBLEM_JSON if problem else DATASET_JSON)
    path.write_text(json.dumps(out, indent=1))
    return path


def write_solution_json(run_dir: Path) -> Path:
    """solution/solution_dataset.json: every possible genotype of every removed person, all families."""
    run_dir = Path(run_dir)
    sol_dir = run_dir / 'solution'
    problem = json.loads((run_dir / 'problem.json').read_text())
    meta = json.loads((sol_dir / 'solution.json').read_text())
    summary = (pd.read_csv(sol_dir / 'solution_summary.csv', dtype={'ID': str}).to_dict('records')
               if (sol_dir / 'solution_summary.csv').exists() else [])

    families = []
    for fam, masked in problem['masked'].items():
        masked = [str(m) for m in masked]
        poss_path = sol_dir / f'{fam}_possible_genotypes.tsv.gz'
        if not poss_path.exists():
            continue
        poss = _read_tsv(poss_path)
        truth: Dict[tuple, str] = {}
        chk_path = sol_dir / f'{fam}_answer_check.tsv.gz'
        if chk_path.exists():
            for r in _read_tsv(chk_path).itertuples(index=False):
                truth[(r.CHROM, r.POS, r.ID)] = _genotype(r.TRUE)
        combos: Dict[tuple, List[dict]] = {}
        comb_path = sol_dir / f'{fam}_combinations.tsv.gz'
        if comb_path.exists():
            for _, r in _read_tsv(comb_path).iterrows():
                combos.setdefault((r['CHROM'], r['POS']), []).append(
                    {p: r[p] for p in masked if p in r and r[p] != '-'})

        variants = []
        for _, r in poss.iterrows():
            key = (r['CHROM'], r['POS'])
            people = {}
            for p in masked:
                cell = r.get(p, '-')
                if cell == '-':  # genotype known at this variant (outside the masked region)
                    continue
                entry = dict(possible=[] if cell == 'CONFLICT' else cell.split('/'),
                             conflict=cell == 'CONFLICT')
                if (key[0], key[1], p) in truth:
                    entry['true'] = truth[(key[0], key[1], p)]
                    entry['true_in_possible'] = entry['true'] in entry['possible']
                people[p] = entry
            v = dict(chrom=key[0], pos=int(key[1]), gene=r['GENE'],
                     combinations=_num(r['COMBINATIONS']), people=people)
            if comb_path.exists():
                v['combination_list'] = combos.get(key, [])
            variants.append(v)
        families.append(dict(
            family=fam, removed_people=masked,
            summary=[{k: (v if k in ('family', 'ID') else _num(v)) for k, v in row.items()}
                     for row in summary if row['family'] == fam],
            n_variants=len(variants), variants=variants))

    out = dict(format_version=FORMAT_VERSION, kind='solution', name=run_dir.name, problem=problem,
               solver=meta, families=families)
    path = sol_dir / SOLUTION_JSON
    path.write_text(json.dumps(out, indent=1))
    return path


def export_folder(run_dir: Path) -> List[Path]:
    """(Re)write every JSON file that applies to this output folder."""
    run_dir = Path(run_dir)
    written = [write_dataset_json(run_dir)]
    if (run_dir / 'problem.json').exists() and (run_dir / 'solution' / 'solution.json').exists():
        written.append(write_solution_json(run_dir))
    return written


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pipeline as P

    args = sys.argv[1:]
    if not args or args == ['-h'] or args == ['--help']:
        print(__doc__)
        sys.exit(0)
    folders = ([r for r in P.list_runs() if (r / 'summary.csv').exists() and not (r / 'FAILED.txt').exists()]
               if args == ['--all'] else [Path(a) for a in args])
    for f in folders:
        try:
            for p in export_folder(f):
                print(f'wrote {p}')
        except Exception as e:  # keep going with the other folders
            print(f'skipped {f}: {e}')
