"""
Pedigree problems: remove the genotypes of chosen people from a generated dataset, then work out every
genotype those people could have had from their relatives' known genotypes.

Inference is exact, one variant at a time, and uses Mendelian inheritance only (no allele
frequencies or linkage). A genotype is "possible" for a person when some assignment of genotypes to
*all* the missing people at that variant is consistent with every parent -> child transmission.
"""

import datetime as dt
import itertools
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import json_export
import pipeline as P

ProgressFn = Callable[[float, str], None]

# Dosage = number of capital (allele 1) letters: AA = 2, Aa = 1, aa = 0.
_GAMETES = {0: (0,), 1: (0, 1), 2: (1,)}
COMPAT = np.zeros((3, 3, 3), dtype=np.int64)  # [parent1, parent2, child] -> 1 if transmission possible
for _f, _m in itertools.product(range(3), repeat=2):
    for _a in _GAMETES[_f]:
        for _b in _GAMETES[_m]:
            COMPAT[_f, _m, _a + _b] = 1
COMPAT_ONE = COMPAT.max(axis=1)              # [parent, child] when the other parent is not in the tree

PROBLEM_FILE = 'problem.json'
SOLUTION_DIR = 'solution'
MAX_EXACT_COUNT = 2 ** 53


# ----------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------

def is_problem(run_dir: Path) -> bool:
    return (run_dir / PROBLEM_FILE).exists()


def source_datasets() -> List[Path]:
    """Finished, genotyped datasets that are not themselves problems."""
    out = []
    for r in P.list_runs():
        if is_problem(r) or not (r / 'summary.csv').exists() or (r / 'FAILED.txt').exists():
            continue
        if 'variants' in pd.read_csv(r / 'summary.csv', nrows=1).columns:
            out.append(r)
    return out


def problem_datasets() -> List[Path]:
    return [r for r in P.list_runs() if is_problem(r)]


def read_problem(run_dir: Path) -> dict:
    return json.loads((run_dir / PROBLEM_FILE).read_text())


def letter_table(fam_dir: Path) -> pd.DataFrame:
    if (fam_dir / 'genotypes_letters.tsv.gz').exists():
        return pd.read_csv(fam_dir / 'genotypes_letters.tsv.gz', sep='\t', dtype=str, keep_default_na=False)
    return P.letter_genotypes(fam_dir / 'genotypes.vcf.gz')


def dosage(cell: str) -> int:
    """'AA' -> 2, 'Aa' -> 1, 'aa' -> 0, anything else (NA, '') -> -1 (missing)."""
    if not isinstance(cell, str) or len(cell) != 2 or not cell.isalpha() or cell[0].lower() != cell[1].lower():
        return -1
    return sum(ch.isupper() for ch in cell)


def letters_for(gene: str, d: int) -> str:
    return {2: gene.upper() * 2, 1: gene.upper() + gene.lower(), 0: gene.lower() * 2}[d]


def read_pedigree(fam_dir: Path) -> nx.DiGraph:
    return nx.read_edgelist(str(fam_dir / 'pedigree.nx'), create_using=nx.DiGraph())


# ----------------------------------------------------------------------------
# making a problem
# ----------------------------------------------------------------------------

def site_in_scope(chrom: str, pos: int, scope: Optional[dict]) -> bool:
    if not scope:
        return True
    return (str(chrom) in {str(c) for c in scope['chromosomes']}
            and (scope.get('start') is None or int(pos) >= int(scope['start']))
            and (scope.get('end') is None or int(pos) <= int(scope['end'])))


def in_scope(chrom: pd.Series, pos: pd.Series, scope: Optional[dict]) -> np.ndarray:
    """Which variants the masking applies to. scope=None means the whole genome."""
    if not scope:
        return np.ones(len(chrom), dtype=bool)
    mask = chrom.astype(str).isin([str(c) for c in scope['chromosomes']]).to_numpy()
    pos = pos.astype(int).to_numpy()
    if scope.get('start') is not None:
        mask &= pos >= int(scope['start'])
    if scope.get('end') is not None:
        mask &= pos <= int(scope['end'])
    return mask


def new_mutation_sites(fam_dir: Path) -> np.ndarray:
    """True for variants that arose as new mutations inside the pedigree (SLiM tick of origin > 1),
    in VCF order. Founder variants are loaded at tick 1. These are the variants that break Mendel's rules."""
    out = subprocess.run(['bcftools', 'query', '-f', '%INFO/TO\n', str(fam_dir / 'genotypes.vcf.gz')],
                         env=P._env(), capture_output=True, text=True)
    return np.array([x.strip() not in ('', '.', '1') for x in out.stdout.splitlines()], dtype=bool)


def _site_is_new_mutation(info: str) -> bool:
    to = next((x[3:] for x in info.split(';') if x.startswith('TO=')), '1')
    return to not in ('', '.', '1')


def _mask_vcf(src: Path, dst: Path, masked: Sequence[str], scope: Optional[dict],
              drop_new_mutations: bool = False) -> None:
    env = P._env()
    reader = subprocess.Popen(['bcftools', 'view', str(src)], env=env, stdout=subprocess.PIPE, text=True)
    writer = subprocess.Popen(['bcftools', 'view', '-Oz', '-o', str(dst), '-'], env=env,
                              stdin=subprocess.PIPE, text=True)
    masked = set(map(str, masked))
    cols = None
    try:
        for line in reader.stdout:
            if line.startswith('##'):
                writer.stdin.write(line)
                continue
            fields = line.rstrip('\n').split('\t')
            if line.startswith('#'):
                cols = [i for i, s in enumerate(fields) if i >= 9 and s in masked]
                writer.stdin.write(line)
                continue
            if drop_new_mutations and _site_is_new_mutation(fields[7]):
                continue
            if site_in_scope(fields[0], fields[1], scope):
                for i in cols:
                    fmt = fields[i].split(':')
                    fmt[0] = './.'
                    fields[i] = ':'.join(fmt)
            writer.stdin.write('\t'.join(fields) + '\n')
    finally:
        writer.stdin.close()
        writer.wait()
        reader.wait()
    if reader.returncode or writer.returncode or not dst.exists():
        raise P.PipelineError(f'Could not write the masked VCF {dst}.')
    subprocess.run(['bcftools', 'index', '-f', str(dst)], env=env, check=True, capture_output=True)


PROBLEM_README = """

Pedigree problem
================
This dataset is a copy of "{source}" in which the genotypes of some people were removed
(see problem.json for who and where). Their entries are ./. in genotypes.vcf.gz and NA in
genotypes_letters.tsv.gz. Everything else (pedigree, profiles, relationships) is unchanged.

problem.json             source dataset, removed people per family, which variants were masked
problem_dataset.json     the whole problem in one JSON file (replaces dataset.json): settings, the
                         problem, and per family every person, their genotype at every variant
                         (null = removed; people have "genotype_removed": true) and relative pairs

solution/  (made by "Solve" in the app; delete this folder to hand out the problem alone)
  solution_summary.csv   one row per removed person: how many variants have 1, 2 or 3 possible
                         genotypes, and (if the source dataset still exists) how often the true
                         genotype is among the possibilities
  family_XXX_possible_genotypes.tsv.gz
                         one row per variant: CHROM, POS, GENE, COMBINATIONS (number of valid
                         joint assignments of all missing genotypes in the family), then one column
                         per removed person listing every genotype they could have, e.g. AA/Aa.
                         "-" = this person's genotype is known at this variant. "CONFLICT" = the
                         known genotypes around this person break Mendel's rules (a new mutation).
  family_XXX_combinations.tsv.gz   (optional) every valid joint assignment per variant
  family_XXX_answer_check.tsv.gz   true genotype next to the possible ones, per variant and person
  solution_dataset.json  the whole solution in one JSON file: per family and variant, every removed
                         person's possible genotypes, the true one, and (if listed) the combinations
"""


def make_problem(source: Path, masks: Dict[str, List[str]], name: str, scope: Optional[dict] = None,
                 drop_new_mutations: bool = True, progress: Optional[ProgressFn] = None) -> Path:
    """Copy `source` into a new dataset with the genotypes of `masks[family]` removed.

    drop_new_mutations: leave out variants that arose inside the pedigree, so every variant left
    follows Mendel's rules and the true genotype is always among the solver's answers."""
    progress = progress or (lambda f, m: None)
    masks = {f: sorted(map(str, ids), key=int) for f, ids in masks.items() if ids}
    if not masks:
        raise P.PipelineError('Choose at least one person whose genotypes should be removed.')
    out = P.unique_run_dir(name)
    out.mkdir(parents=True)
    try:
        for f in source.iterdir():
            if f.is_file() and f.name not in ('log.txt', json_export.DATASET_JSON):
                shutil.copy(f, out / f.name)
        fam_dirs = sorted(d for d in source.iterdir() if d.is_dir() and d.name.startswith('family_'))
        variants, dropped = {}, {}
        for k, fam in enumerate(fam_dirs):
            progress(k / len(fam_dirs), f'Copying {fam.name}')
            dst = out / fam.name
            dst.mkdir()
            ids = masks.get(fam.name, [])
            unknown = set(ids) - set(read_pedigree(fam).nodes)
            if unknown:
                raise P.PipelineError(f'{fam.name} has no person with ID {", ".join(sorted(unknown, key=int))}.')
            new = new_mutation_sites(fam) if drop_new_mutations and (fam / 'genotypes.vcf.gz').exists() else None
            rewrite = bool(ids) or (new is not None and new.any())
            for f in fam.iterdir():
                if f.is_file() and not (rewrite and f.name.startswith('genotypes')):
                    shutil.copy(f, dst / f.name)
            if not rewrite:
                continue
            gt = letter_table(fam)
            if new is not None:
                if len(new) != len(gt):
                    raise P.PipelineError(f'{fam.name}: the VCF and the letters file have different variants.')
                gt = gt[~new].reset_index(drop=True)
                dropped[fam.name] = int(new.sum())
                variants[fam.name] = len(gt)
            rows = in_scope(gt['CHROM'], gt['POS'], scope)
            for i in ids:
                gt.loc[rows, i] = 'NA'
            gt.to_csv(dst / 'genotypes_letters.tsv.gz', sep='\t', index=False)
            _mask_vcf(fam / 'genotypes.vcf.gz', dst / 'genotypes.vcf.gz', ids, scope, new is not None)

        summary = pd.read_csv(out / 'summary.csv')
        if variants:
            summary['variants'] = [variants.get(f, v) for f, v in zip(summary['family'], summary['variants'])]
        summary['removed_people'] = [len(masks.get(f, [])) for f in summary['family']]
        summary.to_csv(out / 'summary.csv', index=False)

        problem = dict(source=source.name, source_path=str(source), scope=scope, masked=masks,
                       new_mutations_dropped=dropped if drop_new_mutations else None,
                       created=dt.datetime.now().isoformat(timespec='seconds'))
        (out / PROBLEM_FILE).write_text(json.dumps(problem, indent=2))
        settings = json.loads((out / 'settings.json').read_text())
        settings['run_name'] = out.name
        settings['problem'] = problem
        (out / 'settings.json').write_text(json.dumps(settings, indent=2))
        readme = (out / 'README.txt').read_text() if (out / 'README.txt').exists() else ''
        (out / 'README.txt').write_text(readme + PROBLEM_README.format(source=source.name))
        json_export.write_dataset_json(out)
        (out / 'log.txt').write_text(
            f'[{dt.datetime.now():%H:%M:%S}] Problem made from {source.name}: removed genotypes of '
            + '; '.join(f'{f}: {", ".join(ids)}' for f, ids in masks.items()) + '\n')
        progress(1.0, 'Done')
        return out
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise


# ----------------------------------------------------------------------------
# exact inference (variable elimination over the pedigree's transmission constraints)
# ----------------------------------------------------------------------------

Factor = Tuple[Tuple[int, ...], np.ndarray]  # (variable ids, table with one axis of size 3 per variable)


def _product(a: Factor, b: Factor) -> Factor:
    vs = tuple(dict.fromkeys(a[0] + b[0]))
    sym = {v: chr(97 + i) for i, v in enumerate(vs)}
    spec = f"{''.join(sym[v] for v in a[0])},{''.join(sym[v] for v in b[0])}->{''.join(sym[v] for v in vs)}"
    return vs, np.einsum(spec, a[1], b[1])


def _combine(factors: List[Factor], out_vars: Sequence[int], counting: bool) -> np.ndarray:
    """Multiply the factors together and sum out every variable not in `out_vars`."""
    acc: Factor = ((), np.ones((), dtype=np.float64 if counting else np.int64))
    for f in factors:
        acc = _product(acc, f)
        if not counting:
            acc = (acc[0], np.minimum(acc[1], 1))
    vs, t = acc
    drop = tuple(i for i, v in enumerate(vs) if v not in out_vars)
    t = t.sum(axis=drop) if drop else t
    kept = [v for v in vs if v in out_vars]
    t = np.transpose(t, [kept.index(v) for v in out_vars]) if kept else t
    return t if counting else np.minimum(t, 1)


def _eliminate(factors: List[Factor], keep: Sequence[int], counting: bool = False) -> np.ndarray:
    """Sum (counting) or OR (boolean) out every variable not in `keep`; returns a table over `keep`."""
    factors = [(vs, t.astype(np.float64) if counting else t) for vs, t in factors]
    keep = list(keep)
    todo = {v for vs, _ in factors for v in vs} - set(keep)
    while todo:
        def cost(v):
            return len({u for vs, _ in factors if v in vs for u in vs})
        v = min(todo, key=cost)
        todo.remove(v)
        touching = [f for f in factors if v in f[0]]
        rest = [f for f in factors if v not in f[0]]
        scope = tuple(sorted({u for vs, _ in touching for u in vs} - {v}))
        rest.append((scope, _combine(touching, scope, counting)))
        factors = rest
    return _combine(factors, keep, counting)


class FamilySolver:
    """Solves one family. Missing genotypes are the variables; each child gives one constraint."""

    def __init__(self, G: nx.DiGraph):
        self.people = sorted(G.nodes, key=int)
        self.index = {p: i for i, p in enumerate(self.people)}
        self.trios = []  # (parent indices, child index)
        for c in self.people:
            ps = tuple(self.index[p] for p in sorted(G.predecessors(c), key=int))
            if ps:
                self.trios.append((ps, self.index[c]))
        self._cache: Dict[Tuple[int, ...], dict] = {}

    def _factors(self, d: Tuple[int, ...]) -> List[Factor]:
        out = []
        for ps, c in self.trios:
            table = COMPAT if len(ps) == 2 else COMPAT_ONE
            vs = ps + (c,)
            idx = tuple(slice(None) if d[v] < 0 else d[v] for v in vs)
            unknown = tuple(v for v in vs if d[v] < 0)
            if unknown:  # constraints among known people only are facts, not information
                out.append((unknown, table[idx]))
        return out

    @staticmethod
    def _components(unknown: List[int], factors: List[Factor]) -> List[Tuple[List[int], List[Factor]]]:
        parent = {v: v for v in unknown}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for vs, _ in factors:
            for v in vs[1:]:
                parent[find(v)] = find(vs[0])
        groups: Dict[int, Tuple[List[int], List[Factor]]] = {}
        for v in unknown:
            groups.setdefault(find(v), ([], []))[0].append(v)
        for f in factors:
            groups[find(f[0][0])][1].append(f)
        return list(groups.values())

    def solve(self, d: Tuple[int, ...], enumerate_up_to: int = 0) -> dict:
        """d: dosage per person (-1 = missing). Returns possible dosages per missing person,
        the number of joint assignments, and (optionally) up to `enumerate_up_to` of them."""
        key = (d, enumerate_up_to)
        if key in self._cache:
            return self._cache[key]
        unknown = [i for i, x in enumerate(d) if x < 0]
        factors = self._factors(d)
        possible, count, comps = {}, 1.0, []
        for vars_, fs in self._components(unknown, factors):
            for v in vars_:
                possible[v] = tuple(int(g) for g in np.flatnonzero(_eliminate(fs, [v])))
            n = float(_eliminate(fs, [], counting=True)) if fs else 3.0 ** len(vars_)
            count *= n
            comps.append((vars_, fs))
        res = dict(possible=possible, count=count, combos=None)
        if enumerate_up_to and count > 0:
            res['combos'] = self._enumerate(comps, possible, enumerate_up_to)
        self._cache[key] = res
        return res

    @staticmethod
    def _enumerate(comps, possible, limit) -> List[Dict[int, int]]:
        per_comp = []
        for vars_, fs in comps:
            sols, assign = [], {}
            checks = {v: [f for f in fs if v in f[0] and all(u in vars_[:vars_.index(v) + 1] for u in f[0])]
                      for v in vars_}

            def walk(k):
                if len(sols) >= limit:
                    return
                if k == len(vars_):
                    sols.append(dict(assign))
                    return
                v = vars_[k]
                for g in possible[v]:
                    assign[v] = g
                    if all(t[tuple(assign[u] for u in vs)] for vs, t in checks[v]):
                        walk(k + 1)
                del assign[v]
            walk(0)
            per_comp.append(sols)
        out = []
        for parts in itertools.product(*per_comp):
            merged = {}
            for p in parts:
                merged.update(p)
            out.append(merged)
            if len(out) >= limit:
                break
        return out


# ----------------------------------------------------------------------------
# solving a problem dataset
# ----------------------------------------------------------------------------

def _fmt_count(n: float) -> str:
    return str(int(n)) if n < MAX_EXACT_COUNT else f'{n:.3e}'


def solve_problem(run_dir: Path, list_combinations: bool = False, max_combinations: int = 1000,
                  progress: Optional[ProgressFn] = None) -> Path:
    progress = progress or (lambda f, m: None)
    problem = read_problem(run_dir)
    sol_dir = run_dir / SOLUTION_DIR
    if sol_dir.exists():
        shutil.rmtree(sol_dir)
    sol_dir.mkdir()
    source = Path(problem.get('source_path', ''))
    if not (source / 'summary.csv').exists():
        source = P.OUTPUTS / problem['source']
    has_truth = (source / 'summary.csv').exists()

    summary_rows = []
    fams = list(problem['masked'].items())
    for k, (fam, masked) in enumerate(fams):
        fam_dir = run_dir / fam
        solver = FamilySolver(read_pedigree(fam_dir))
        gt = letter_table(fam_dir)
        people = solver.people
        D = np.column_stack([gt[p].map(dosage).to_numpy() for p in people])
        truth = None
        if has_truth:
            t = letter_table(source / fam)
            if problem.get('new_mutations_dropped') is not None:
                new = new_mutation_sites(source / fam)
                if len(new) == len(t):
                    t = t[~new].reset_index(drop=True)
            if len(t) == len(gt) and (t['POS'].to_numpy() == gt['POS'].to_numpy()).all():
                truth = np.column_stack([t[p].map(dosage).to_numpy() for p in masked])

        cols = {p: [] for p in masked}
        counts, combo_rows, check_rows = [], [], []
        n_sites = len(gt)
        for s in range(n_sites):
            if s % 200 == 0:
                progress((k + s / max(n_sites, 1)) / len(fams), f'Solving {fam}: variant {s + 1:,} of {n_sites:,}')
            gene = gt['GENE'].iat[s]
            res = solver.solve(tuple(int(x) for x in D[s]), max_combinations if list_combinations else 0)
            counts.append(_fmt_count(res['count']))
            for j, p in enumerate(masked):
                i = solver.index[p]
                if i not in res['possible']:
                    cols[p].append('-')
                    continue
                opts = sorted(res['possible'][i], reverse=True)
                cell = '/'.join(letters_for(gene, g) for g in opts) if opts else 'CONFLICT'
                cols[p].append(cell)
                if truth is not None:
                    true_d = int(truth[s, j])
                    check_rows.append((gt['CHROM'].iat[s], gt['POS'].iat[s], gene, p,
                                       letters_for(gene, true_d) if true_d >= 0 else 'NA', cell,
                                       len(opts), true_d in opts))
            if res['combos']:
                order = [solver.index[p] for p in masked]
                for c, combo in enumerate(res['combos'], start=1):
                    combo_rows.append([gt['CHROM'].iat[s], gt['POS'].iat[s], gene, c] +
                                      [letters_for(gene, combo[i]) if i in combo else '-' for i in order])

        table = gt[['CHROM', 'POS', 'GENE']].copy()
        table['COMBINATIONS'] = counts
        for p in masked:
            table[p] = cols[p]
        table.to_csv(sol_dir / f'{fam}_possible_genotypes.tsv.gz', sep='\t', index=False)
        if list_combinations:
            pd.DataFrame(combo_rows, columns=['CHROM', 'POS', 'GENE', 'COMBINATION', *masked]).to_csv(
                sol_dir / f'{fam}_combinations.tsv.gz', sep='\t', index=False)
        check = pd.DataFrame(check_rows, columns=['CHROM', 'POS', 'GENE', 'ID', 'TRUE', 'POSSIBLE',
                                                  'N_POSSIBLE', 'TRUE_IN_POSSIBLE'])
        if truth is not None:
            check.to_csv(sol_dir / f'{fam}_answer_check.tsv.gz', sep='\t', index=False)

        for p in masked:
            c = table[p]
            solved = c[c != '-']
            n_opts = solved.map(lambda x: 0 if x == 'CONFLICT' else x.count('/') + 1)
            row = dict(family=fam, ID=p, masked_variants=len(solved),
                       one_possible=int((n_opts == 1).sum()), two_possible=int((n_opts == 2).sum()),
                       three_possible=int((n_opts == 3).sum()), conflicts=int((n_opts == 0).sum()))
            if truth is not None:
                mine = check[check['ID'] == p]
                row['true_in_possible_pct'] = round(100 * mine['TRUE_IN_POSSIBLE'].mean(), 2) if len(mine) else None
            summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(sol_dir / 'solution_summary.csv', index=False)
    (sol_dir / 'solution.json').write_text(json.dumps(dict(
        solved=dt.datetime.now().isoformat(timespec='seconds'), list_combinations=list_combinations,
        max_combinations_per_variant=max_combinations if list_combinations else None,
        answer_check=has_truth, method='exact Mendelian consistency, one variant at a time'), indent=2))
    json_export.write_solution_json(run_dir)
    progress(1.0, 'Done')
    return sol_dir
