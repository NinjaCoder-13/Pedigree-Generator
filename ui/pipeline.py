"""
Dataset pipeline behind the py_ped_sim web UI (ui/app.py).

It chains the repo's existing tools (sim_pedigree, family broadening,
msprime founders, filter_vcf, SLiM genome simulation, enur_fam) and writes
everything for one run into outputs/<run_name>/.

Every job runs in a temporary directory with no spaces in its path, because
several repo scripts build unquoted shell commands. Finished files are then
copied into outputs/, so the repo can live anywhere.
"""

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / 'outputs'
SIBSHIP_FILE = REPO / 'scripts' / 'ipumps_sibship_dist.txt'

RATES = ['1e-7', '1e-8', '1e-9']

PRESETS = {
    'Tiny': dict(years=[1980, 2011], size=(4, 15),
                 blurb='3 generations: a couple, their children and grandchildren (about 5-15 people)'),
    'Small': dict(years=[1950, 1980, 2011], size=(15, 60),
                  blurb='4 generations: down to great-grandchildren (about 15-60 people)'),
    'Medium': dict(years=[1900, 1930, 1960, 1990], size=(60, 200),
                   blurb='5 generations: a large extended family (about 60-200 people)'),
    'Large': dict(years=[1870, 1900, 1930, 1960, 1990], size=(150, 600),
                  blurb='6 generations: a very large extended family (about 150-600 people)'),
}

LogFn = Callable[[str], None]


class PipelineError(Exception):
    pass


@dataclass
class Region:
    """One chromosome / genomic region to simulate."""
    name: str = '1'
    length_bp: int = 1_000_000
    recomb_rate: str = '1e-8'


@dataclass
class RunConfig:
    run_name: str = 'my_dataset'
    n_families: int = 1
    seed: int = 1
    years: List[int] = field(default_factory=lambda: [1950, 1980, 2011])
    min_size: int = 15
    max_size: int = 60
    max_attempts: int = 200
    broaden: bool = False
    half_sibs: bool = False
    half_sib_prob: float = 0.10
    genotypes: bool = True
    founder_source: str = 'simulate'  # 'simulate' or 'upload'
    regions: List[Region] = field(default_factory=lambda: [Region()])
    uploaded_vcfs: List[str] = field(default_factory=list)
    ne: int = 10000
    mutation_rate: str = '1e-8'
    new_mutations: bool = False  # new mutations while DNA is passed down the family (breaks Mendel)
    relationships: bool = True
    ped: bool = True
    picture: bool = True


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def sibship_table() -> pd.DataFrame:
    return pd.read_csv(SIBSHIP_FILE, sep='\t')


def available_years() -> List[int]:
    return sorted(sibship_table()['Year'].astype(int).tolist())


def safe_name(name: str) -> str:
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip()).strip('_')
    return name or 'dataset'


def unique_run_dir(name: str) -> Path:
    base = OUTPUTS / safe_name(name)
    path, i = base, 2
    while path.exists():
        path = Path(f'{base}_{i}')
        i += 1
    return path


def list_runs() -> List[Path]:
    if not OUTPUTS.exists():
        return []
    runs = [p for p in OUTPUTS.iterdir() if p.is_dir() and (p / 'settings.json').exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _env() -> Dict[str, str]:
    # Make nested `python`, `slim`, `bcftools` calls resolve to this environment.
    env = dict(os.environ)
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    return env


def _run(cmd: List[str], step: str, log: LogFn, expect: Optional[Path] = None) -> str:
    cmd = [str(c) for c in cmd]
    if cmd[0] == 'python':
        cmd[0] = sys.executable
    proc = subprocess.run(cmd, cwd=REPO, env=_env(), capture_output=True, text=True)
    out = (proc.stdout or '') + (proc.stderr or '')
    # Many repo scripts exit 0 on failure, so check for the expected output too.
    if proc.returncode != 0 or (expect is not None and not Path(expect).exists()):
        tail = '\n'.join(out.strip().splitlines()[-15:])
        raise PipelineError(f'{step} failed.\n\n{tail}')
    return out


def _run_ped_sim(args: List, step: str, log: LogFn, expect: Optional[Path] = None) -> str:
    return _run(['python', REPO / 'run_ped_sim.py'] + list(args), step, log, expect)


# ----------------------------------------------------------------------------
# pedigree structure
# ----------------------------------------------------------------------------

def _simulate_structure(cfg: RunConfig, seed: int, work: Path, log: LogFn) -> Tuple[Path, Path]:
    years = [str(y) for y in cfg.years]
    if cfg.broaden:
        out = work / 'ped'
        _run_ped_sim(['-t', 'run_full_family_broadening', '-y', *years,
                      '-mo', work / 'main', '-o', out],
                     'Family broadening', log, expect=Path(f'{out}.nx'))
    else:
        out = work / 'ped'
        _run(['python', REPO / 'scripts' / 'sim_pedigree.py', '-y', *years, '-s', seed,
              '-c', SIBSHIP_FILE, '-o', out],
             'Pedigree simulation', log, expect=Path(f'{out}.nx'))
    return Path(f'{out}.nx'), Path(f'{out}_profiles.txt')


def _add_half_sibs(G: nx.DiGraph, prof: pd.DataFrame, prob: float,
                   rng: np.random.Generator) -> Tuple[nx.DiGraph, pd.DataFrame, int]:
    """
    Misattributed paternity: each child whose mother belongs to the family (has parents
    in the tree) gets, with probability `prob`, a new father from outside the family.
    That child becomes a half-sibling of the mother's other children.

    Used instead of scripts/sim_map.py, which can pick a married-in man of another
    branch as the new father and leave children with no blood link to the family.
    """
    sex = dict(zip(prof['ID'], prof['Sex']))
    gen = dict(zip(prof['ID'], prof['Gen']))
    next_id = max(int(n) for n in G.nodes) + 1
    events = 0
    for child in sorted(G.nodes, key=int):
        ps = list(G.predecessors(child))
        if len(ps) != 2 or rng.random() >= prob:
            continue
        mom = next((p for p in ps if sex.get(p) == 'female'), None)
        dad = next((p for p in ps if p != mom), None)
        if mom is None or G.in_degree(mom) == 0:
            continue
        new = str(next_id)
        next_id += 1
        G.remove_edge(dad, child)
        G.add_edge(new, child)
        sex[new], gen[new] = 'male', gen.get(mom)
        events += 1
    # A father who lost all his children is no longer part of the family.
    G.remove_nodes_from([n for n in list(G.nodes) if G.in_degree(n) == 0 and G.out_degree(n) == 0])
    nodes = sorted(G.nodes, key=int)
    prof = pd.DataFrame({'ID': nodes, 'Sex': [sex[n] for n in nodes],
                         'Gen': pd.array([gen.get(n) for n in nodes], dtype='Int64')})
    return G, prof, events


def _complete_pedigree(nx_path: Path, prof_path: Path) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Make the pedigree simulation-ready:
    - every child gets two parents (a missing one is added as a new founder,
      shared by all children of that single parent so they stay full siblings)
    - every person has a sex and generation in the profiles table
    """
    G = nx.read_edgelist(str(nx_path), create_using=nx.DiGraph())
    prof = pd.read_csv(prof_path, sep='\t', dtype={'ID': str})
    prof['Gen'] = pd.to_numeric(prof['Gen'], errors='coerce')
    prof = prof[prof['ID'].isin(G.nodes)]
    sex = dict(zip(prof['ID'], prof['Sex']))
    gen = dict(zip(prof['ID'], prof['Gen']))
    other = {'male': 'female', 'female': 'male'}

    next_id = max(int(n) for n in G.nodes) + 1
    new_partner = {}
    for child in sorted(G.nodes, key=int):
        parents = list(G.predecessors(child))
        if len(parents) != 1:
            continue
        p = parents[0]
        if p not in new_partner:
            new_partner[p] = str(next_id)
            next_id += 1
        G.add_edge(new_partner[p], child)

    # Fill in sex/generation for people without a profile (added fathers/partners).
    for n in sorted(G.nodes, key=int):
        if n in sex:
            continue
        co_parents = {q for c in G.successors(n) for q in G.predecessors(c) if q != n and q in sex}
        q = next(iter(co_parents), None)
        sex[n] = other.get(sex.get(q), 'male')
        gen[n] = gen.get(q, np.nan)

    nodes = sorted(G.nodes, key=int)
    prof = pd.DataFrame({'ID': nodes, 'Sex': [sex[n] for n in nodes],
                         'Gen': pd.array([gen.get(n, np.nan) for n in nodes], dtype='Float64').astype('Int64')})
    return G, prof


def generation_layers(G: nx.DiGraph, prof: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """
    Generation index per person (0 = founding couple). Uses the birth-cohort year in the
    profiles when available, so married-in partners sit in their spouse's generation.
    """
    layer = {}
    gen = {}
    if prof is not None:
        gen = {str(i): g for i, g in zip(prof['ID'], pd.to_numeric(prof['Gen'], errors='coerce')) if pd.notna(g)}
    if gen:
        rank = {y: i + 1 for i, y in enumerate(sorted(set(gen.values())))}
        layer = {n: rank[gen[n]] for n in G.nodes if n in gen}
        for n in G.nodes:
            if n not in layer:
                kids = [layer[k] for k in G.successors(n) if k in layer]
                layer[n] = min(kids) - 1 if kids else 0
        if all(layer[v] > layer[u] for u, v in G.edges):
            return layer
    # Fallback: derive from the graph alone.
    layer = {}
    for n in nx.topological_sort(G):
        ps = list(G.predecessors(n))
        layer[n] = 0 if not ps else max(layer[p] for p in ps) + 1
    for n in G.nodes:
        kids = list(G.successors(n))
        if not list(G.predecessors(n)) and kids:
            layer[n] = min(layer[k] for k in kids) - 1
    return layer


def draw_pedigree(G: nx.DiGraph, prof: pd.DataFrame, path, title: str = '',
                  genotypes: Optional[Dict[str, str]] = None, highlight: Optional[set] = None,
                  fmt: Optional[str] = None) -> None:
    """Classic pedigree chart: partners side by side, a bar from each couple down to their children.

    `genotypes` (person ID -> letter genotype such as 'Aa') is written under each person.
    `highlight` (person IDs) are drawn grey, e.g. people whose genotypes were removed.
    `path` can be a file path or a binary buffer. `fmt` is 'png' or 'svg' (default: from the file
    suffix, else png). An SVG is drawn at full scale however big the family is, so it stays sharp
    and readable when zoomed; a PNG is capped in width and shrinks labels for big families.
    """
    if fmt is None:
        fmt = 'svg' if str(getattr(path, 'suffix', '')).lower() == '.svg' else 'png'
    scalable = fmt == 'svg'
    sex = dict(zip(prof['ID'].astype(str), prof['Sex']))
    layer = generation_layers(G, prof)
    n_layers = max(layer.values()) + 1
    rows = {l: [] for l in range(n_layers)}
    for n in G.nodes:
        rows[layer[n]].append(n)

    def partners(n):
        return sorted({q for k in G.successors(n) for q in G.predecessors(k) if q != n}, key=int)

    # Lay out each generation top-down: blood relatives ordered under their parents,
    # each followed directly by the founders who married in.
    pos = {}
    for l in range(n_layers):
        here = set(rows[l])
        married_in = {n for n in here if l > 0 and G.in_degree(n) == 0}
        core = [n for n in here if n not in married_in]

        def key(n):
            ps = [pos[p][0] for p in G.predecessors(n) if p in pos]
            return (np.mean(ps) if ps else 0.0, tuple(sorted(G.predecessors(n), key=int)), int(n))

        ordered, placed = [], set()
        for n in sorted(core, key=key):
            if n in placed:
                continue
            ordered.append(n)
            placed.add(n)
            for q in partners(n):
                if q in here and q not in placed and (q in married_in or l == 0):
                    ordered.append(q)
                    placed.add(q)
        ordered += sorted(here - placed, key=int)

        # Aim each child at its parents' midpoint, then push apart to keep one unit of spacing.
        want = []
        for i, n in enumerate(ordered):
            ps = [pos[p][0] for p in G.predecessors(n) if p in pos]
            want.append(float(np.mean(ps)) if ps else (want[-1] + 1 if want else 0.0))
        xs = []
        for w in want:
            xs.append(w if not xs else max(w, xs[-1] + 1))
        shift = float(np.mean(np.array(want) - np.array(xs)))
        for n, x in zip(ordered, xs):
            pos[n] = (x + shift, -l)

    big = not scalable and G.number_of_nodes() > 80
    x_min = min(x for x, _ in pos.values())
    x_max = max(x for x, _ in pos.values())
    width = x_max - x_min + 1
    fig_w = max(6, width * 0.7) if scalable else min(max(6, width * 0.6), 60)
    fig, ax = plt.subplots(figsize=(fig_w, 1.7 * n_layers + 1.2))
    line = dict(color='#8a94a6', lw=0.8, zorder=1, solid_capstyle='butt')

    sibships = {}
    for c in G.nodes:
        ps = tuple(sorted(G.predecessors(c), key=int))
        if ps:
            sibships.setdefault(ps, []).append(c)
    bar_count = {}
    for ps, kids in sorted(sibships.items(), key=lambda kv: min(pos[k][0] for k in kv[1])):
        px = [pos[p][0] for p in ps]
        py = [pos[p][1] for p in ps]
        kx = sorted(pos[k][0] for k in kids)
        ky = pos[kids[0]][1]
        if len(set(py)) == 1 and all(pos[k][1] == ky for k in kids):
            top = py[0]
            if len(ps) == 2:
                ax.plot(px, [top, top], **line)                          # partnership line
            k = bar_count.get(top, 0)
            bar_count[top] = k + 1
            mid_x, bar_y = float(np.mean(px)), top - 0.3 - 0.2 * (k % 3)  # staggered so families stay distinct
            ax.plot([mid_x, mid_x], [top, bar_y], **line)                 # drop to sibship bar
            ax.plot([min(kx + [mid_x]), max(kx + [mid_x])], [bar_y, bar_y], **line)
            for x in kx:
                ax.plot([x, x], [bar_y, ky], **line)                      # down to each child
        else:  # parents in different generations: draw direct lines
            for p in ps:
                for k in kids:
                    ax.plot([pos[p][0], pos[k][0]], [pos[p][1], pos[k][1]], **line)

    highlight = {str(n) for n in highlight or ()}
    for s, marker, color in [('male', 's', '#4c78a8'), ('female', 'o', '#e45756')]:
        for grey in (False, True):
            pts = [pos[n] for n in G.nodes if sex.get(n) == s and (str(n) in highlight) == grey]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, marker=marker, s=30 if big else 300, c='#9ca3af' if grey else color,
                           edgecolors='#111827' if grey else 'white', linewidths=1.5 if grey else 1, zorder=2,
                           label=None if grey else s.capitalize())
    if highlight:
        ax.scatter([], [], marker='s', s=60, c='#9ca3af', edgecolors='#111827', label='Genotypes removed')
    if not big:
        for n, (x, y) in pos.items():
            ax.text(x, y, n, ha='center', va='center', fontsize=7, color='white', zorder=3, fontweight='bold')
    if genotypes:
        for n, (x, y) in pos.items():
            gt = genotypes.get(str(n))
            if gt:
                ax.text(x, y - (0.12 if big else 0.2), gt, ha='center', va='top', fontsize=5 if big else 8,
                        color='#1f2937', zorder=3, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    ax.set_yticks([-l for l in range(n_layers)])
    ax.set_yticklabels([f'Gen {l + 1}' for l in range(n_layers)])
    ax.set_xticks([])
    ax.set_xlim(x_min - 0.7, x_max + 0.7)
    ax.set_ylim(-n_layers + 0.5, 0.5)
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(False)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.0), ncol=3, frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11, loc='left')
    fig.tight_layout()
    with plt.rc_context({'svg.fonttype': 'none'}):   # keep labels as real text: crisp and searchable
        fig.savefig(path, dpi=130, format=fmt)
    plt.close(fig)


def _write_ped(G: nx.DiGraph, prof: pd.DataFrame, fam_id: str, path: Path) -> pd.DataFrame:
    """Standard 6-column PED/FAM: FID IID father mother sex(1=M,2=F) phenotype(-9)."""
    sex = dict(zip(prof['ID'].astype(str), prof['Sex']))
    rows = []
    for n in sorted(G.nodes, key=int):
        ps = list(G.predecessors(n))
        dad = next((p for p in ps if sex.get(p) == 'male'), '0')
        mom = next((p for p in ps if p != dad), '0')
        rows.append([fam_id, n, dad, mom, 1 if sex.get(n) == 'male' else 2, -9])
    df = pd.DataFrame(rows, columns=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENO'])
    df.to_csv(path, sep=' ', index=False, header=False)
    return df


def _make_family(cfg: RunConfig, fam_seed: int, work: Path, log: LogFn) -> Tuple[nx.DiGraph, pd.DataFrame, int]:
    """Simulate pedigrees until one lands inside the requested size range."""
    rng = np.random.default_rng(fam_seed)
    for attempt in range(1, cfg.max_attempts + 1):
        seed = int(rng.integers(1, 2**31 - 1))
        for f in work.glob('ped*'):
            f.unlink()
        nx_path, prof_path = _simulate_structure(cfg, seed, work, log)
        G, prof = _complete_pedigree(nx_path, prof_path)
        if cfg.half_sibs:
            G, prof, _ = _add_half_sibs(G, prof, cfg.half_sib_prob, np.random.default_rng(seed))
        if cfg.min_size <= G.number_of_nodes() <= cfg.max_size:
            return G, prof, attempt
    raise PipelineError(
        f'Could not get a family with {cfg.min_size}-{cfg.max_size} people in {cfg.max_attempts} tries. '
        'Widen the size range or pick a different family size.')


# ----------------------------------------------------------------------------
# genomes
# ----------------------------------------------------------------------------

def _samples(vcf: Path) -> List[str]:
    out = subprocess.run(['bcftools', 'query', '-l', str(vcf)], env=_env(), capture_output=True, text=True)
    return out.stdout.split()


def _first_chrom(vcf: Path) -> str:
    out = subprocess.run(f"bcftools view -H '{vcf}' | head -1 | cut -f1", shell=True,
                         env=_env(), capture_output=True, text=True)
    return out.stdout.strip() or '1'


def _filter_for_slim(vcf: Path, prefix: Path, log: LogFn) -> Path:
    """
    Same filters as util.filter_vcf_for_slim (bi-allelic SNPs, MAC > 0, no missing
    genotypes, no duplicated positions, INFO/AA = REF), but the AA annotation is done
    here because `bcftools annotate` from the env's bcftools 1.9 segfaults on some inputs.
    """
    snps = Path(f'{prefix}_snps.vcf')
    _run(['bash', '-c', 'bcftools view -m2 -M2 -v snps "$1" -Ou | bcftools filter -e "MAC == 0" -Ou '
                        '| bcftools view -i "F_MISSING == 0" -Ov -o "$2"', '_', vcf, snps],
         'VCF filtering', log, expect=snps)

    with open(snps) as f:
        lines = f.readlines()
    pos_count = {}
    for l in lines:
        if not l.startswith('#'):
            key = tuple(l.split('\t', 2)[:2])
            pos_count[key] = pos_count.get(key, 0) + 1

    out = Path(f'{prefix}_slim_fil.vcf')
    with open(out, 'w') as f:
        for l in lines:
            if l.startswith('##INFO=<ID=AA,'):
                continue
            if l.startswith('#CHROM'):
                f.write('##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">\n')
            if l.startswith('#'):
                f.write(l)
                continue
            cols = l.rstrip('\n').split('\t')
            if pos_count[(cols[0], cols[1])] > 1 or int(cols[1]) < 1:  # VCF is 1-based; msprime can emit 0
                continue
            info = [x for x in cols[7].split(';') if x not in ('.', '') and not x.startswith('AA=')]
            cols[7] = ';'.join(info + [f'AA={cols[3].upper()}'])
            f.write('\t'.join(cols) + '\n')
    snps.unlink()
    return out


def _prepare_founder_pools(cfg: RunConfig, n_founders_total: int, seed: int, work: Path,
                           log: LogFn) -> List[Tuple[str, Path]]:
    pools = []
    if cfg.founder_source == 'simulate':
        n_pool = max(n_founders_total, 20)
        for k, reg in enumerate(cfg.regions):
            log(f'Simulating {n_pool} founder genomes for region "{reg.name}" '
                f'({reg.length_bp:,} bp) with msprime...')
            prefix = work / f'pool{k}'
            _run(['python', REPO / 'scripts' / 'sim_fam_msprime.py', '-Nf', n_pool, '-Ne', cfg.ne,
                  '-l', reg.length_bp, '-s', seed + k, '-mu', cfg.mutation_rate, '-r', reg.recomb_rate,
                  '-o', prefix], 'Founder genome simulation', log, expect=Path(f'{prefix}.vcf'))
            pools.append((reg.name, _filter_for_slim(Path(f'{prefix}.vcf'), prefix, log)))
    else:
        for k, src in enumerate(cfg.uploaded_vcfs):
            src = Path(src)
            local = work / f'upload{k}{"".join(src.suffixes[-2:]) if src.suffix == ".gz" else ".vcf"}'
            shutil.copy(src, local)
            name = _first_chrom(local)
            log(f'Preparing your VCF "{src.name}" (chromosome {name}) for SLiM...')
            pools.append((name, _filter_for_slim(local, work / f'pool{k}', log)))
    return pools


def _simulate_genomes(G: nx.DiGraph, fam_dir: Path, work: Path, pools: List[Tuple[str, Path]],
                      founder_samples: List[str], cfg: RunConfig, fam_seed: int, log: LogFn) -> int:
    founders = sorted([n for n in G.nodes if G.in_degree(n) == 0], key=int)
    ped_nx = work / 'pedigree.nx'
    nx.write_edgelist(G, str(ped_nx))
    exact = work / 'exact_founders.txt'
    exact.write_text(''.join(f'{s} {f}\n' for s, f in zip(founder_samples, founders)))

    region_vcfs = []
    for k, (name, pool) in enumerate(pools):
        out = work / f'genome{k}'
        _run_ped_sim(['-t', 'sim_genomes_exact', '-n', ped_nx, '-v', pool, '-e', exact,
                      '-mu', cfg.mutation_rate if cfg.new_mutations else '0',
                      '-r', cfg.regions[k].recomb_rate if cfg.founder_source == 'simulate' else '1e-8',
                      '-s', fam_seed + 1000 * k, '-o', out],
                     f'Genome simulation (region {name})', log, expect=Path(f'{out}_genomes.vcf'))
        chrom_map = work / f'chrom{k}.txt'
        chrom_map.write_text(f'1\t{name}\n')
        renamed = work / f'genome{k}.vcf.gz'
        _run(['bcftools', 'annotate', '--rename-chrs', chrom_map, f'{out}_genomes.vcf', '-Oz', '-o', renamed],
             'Renaming chromosomes', log, expect=renamed)
        _run(['bcftools', 'index', '-f', renamed], 'Indexing VCF', log)
        region_vcfs.append(renamed)

    final = fam_dir / 'genotypes.vcf.gz'
    if len(region_vcfs) == 1:
        shutil.copy(region_vcfs[0], final)
    else:
        _run(['bcftools', 'concat', *region_vcfs, '-Oz', '-o', final], 'Combining regions', log, expect=final)
    _run(['bcftools', 'index', '-f', final], 'Indexing VCF', log)
    n = subprocess.run(['bcftools', 'index', '-n', str(final)], env=_env(), capture_output=True, text=True)
    return int(n.stdout.strip() or 0)


_DOSAGE = {'1|1': 2, '1/1': 2, '1|0': 1, '0|1': 1, '1/0': 1, '0/1': 1, '0|0': 0, '0/0': 0}


def letter_genotypes(vcf: Path, seed: int = 0, limit: Optional[int] = None) -> pd.DataFrame:
    """VCF genotypes as letters: 1|1 -> AA, 1|0 = 0|1 -> Aa, 0|0 -> aa (NA if missing).

    The simulation is unchanged; allele 1 is shown as the dominant (capital) letter and
    allele 0 as the recessive (small) one. Each variant gets a random letter: the alphabet
    is shuffled and repeated, so any 26 consecutive variants have different letters.
    """
    samples = _samples(vcf)
    cmd = ['bcftools', 'query', '-f', '%CHROM\t%POS[\t%GT]\n', str(vcf)]
    proc = subprocess.Popen(cmd, env=_env(), stdout=subprocess.PIPE, text=True)
    try:
        df = pd.read_csv(proc.stdout, sep='\t', header=None, names=['CHROM', 'POS', *samples],
                         dtype=str, nrows=limit)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['CHROM', 'POS', *samples])
    finally:
        proc.kill()
        proc.wait()
    rng = np.random.default_rng(seed)
    alphabet = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    reps = -(-len(df) // 26)
    big = np.concatenate([rng.permutation(alphabet) for _ in range(reps)])[:len(df)] if reps else alphabet[:0]
    small = np.char.lower(big)
    hom_dom, het, hom_rec = np.char.add(big, big), np.char.add(big, small), np.char.add(small, small)
    out = df[['CHROM', 'POS']].copy()
    out.insert(2, 'GENE', big)
    for s in samples:
        d = df[s].map(_DOSAGE).to_numpy()
        out[s] = np.select([d == 2, d == 1, d == 0], [hom_dom, het, hom_rec], 'NA')
    return out


def _relationships(G: nx.DiGraph, fam_name: str, work: Path, log: LogFn) -> pd.DataFrame:
    ped_nx = work / 'pedigree.nx'
    nx.write_edgelist(G, str(ped_nx))
    out = work / 'rel'
    _run(['python', REPO / 'scripts' / 'enur_fam.py', '-n', ped_nx, '-o', out],
         'Relationship table', log, expect=Path(f'{out}_rel.csv'))
    rel = pd.read_csv(f'{out}_rel.csv', dtype=str)
    rel['Fam_ID'] = fam_name
    rel['Relationship'] = [relationship_name(int(mc), int(gd), rt)
                           for mc, gd, rt in zip(rel['MC'], rel['GD'], rel['RT'])]
    return rel


_ORDINAL = ['Zeroth', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth']
_TIMES = {1: 'Once', 2: 'Twice', 3: 'Three Times', 4: 'Four Times', 5: 'Five Times'}


def relationship_name(mc: int, gd: int, rt: str) -> str:
    """Standard genealogical name from meiosis count, generation gap and direct/full/half."""
    if rt == 'direct':
        if mc == 1:
            return 'Parent-Child'
        return 'Great-' * (mc - 2) + 'Grandparent'
    near, far = (mc - gd) // 2, (mc + gd) // 2  # steps from each person up to the common ancestor(s)
    kind = {'full': 'Full ', 'half': 'Half '}.get(rt, '')
    if near == 1 and far == 1:
        return f'{kind}Sibling'
    if near == 1:
        return f'{kind}' + 'Great-' * (far - 2) + 'Aunt/Uncle'
    degree = _ORDINAL[near - 1] if near - 1 < len(_ORDINAL) else f'{near - 1}th'
    name = f'{kind}{degree} Cousin'
    if gd:
        name += f' {_TIMES.get(gd, f"{gd} Times")} Removed'
    return name


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------

README_TEXT = """\
Pedigree dataset generated with py_ped_sim
==========================================

summary.csv              one row per family (people, founders, generations, variants)
settings.json            the exact settings used (re-use the seed to reproduce)
all_families.ped         every family in one PED/FAM file (FID IID father mother sex phenotype)
all_relationships.csv    every pair of relatives, all families
log.txt                  what happened during the run
dataset.json             everything above in one JSON file: settings, summary and, per family,
                         every person (sex, birth year, generation, father, mother, children),
                         their genotype at every variant (AA / Aa / aa) and the relative pairs

family_XXX/
  pedigree.nx            the family tree (networkx edge list: parent child)
  profiles.txt           ID, Sex, Gen (birth-cohort year; NA if unknown)
  pedigree.ped           this family in PED/FAM format
  relationships.csv      ID1, ID2, MC (meioses), GD (generation gap), RT (direct/full/half),
                         RC (original py_ped_sim label), Relationship (standard name)
  founder_genomes.txt    which VCF sample each founder's DNA came from
  genotypes.vcf.gz       phased genotypes for every person, one chromosome per region
  genotypes_letters.tsv.gz  the same genotypes as letters, one row per variant:
                         CHROM, POS, GENE (random letter), then one column per person with
                         AA = 1|1 (both dominant), Aa = 1|0 or 0|1, aa = 0|0 (both recessive)
  pedigree.png           picture of the family (squares = male, circles = female)

Sex codes in .ped files: 1 = male, 2 = female. Missing parent = 0.
"""


def run_pipeline(cfg: RunConfig, log: LogFn = print, progress: Callable[[float, str], None] = None) -> Path:
    progress = progress or (lambda f, m: None)
    started = time.time()
    run_dir = unique_run_dir(cfg.run_name)
    run_dir.mkdir(parents=True)
    log_lines = []

    def _log(msg):
        line = f'[{dt.datetime.now():%H:%M:%S}] {msg}'
        log_lines.append(line)
        log(line)

    tmp_root = Path(tempfile.mkdtemp(prefix='pedsim_'))
    try:
        settings = asdict(cfg)
        settings['uploaded_vcfs'] = [Path(p).name for p in cfg.uploaded_vcfs]
        settings['created'] = dt.datetime.now().isoformat(timespec='seconds')
        (run_dir / 'settings.json').write_text(json.dumps(settings, indent=2))
        (run_dir / 'README.txt').write_text(README_TEXT)

        master = np.random.default_rng(cfg.seed)
        fam_seeds = [int(s) for s in master.integers(1, 2**31 - 1, size=cfg.n_families)]
        width = max(3, len(str(cfg.n_families)))
        steps = cfg.n_families * (2 if cfg.genotypes else 1) + (1 if cfg.genotypes else 0)
        done = 0

        # 1) pedigrees
        families = []
        for i, fs in enumerate(fam_seeds, start=1):
            name = f'family_{i:0{width}d}'
            work = tmp_root / name
            work.mkdir()
            progress(done / steps, f'Building {name} ({i}/{cfg.n_families})')
            G, prof, tries = _make_family(cfg, fs, work, _log)
            founders = [n for n in G.nodes if G.in_degree(n) == 0]
            _log(f'{name}: {G.number_of_nodes()} people, {len(founders)} founders'
                 + (f' (re-drew {tries - 1}x to fit the size range)' if tries > 1 else ''))
            fam_dir = run_dir / name
            fam_dir.mkdir()
            nx.write_edgelist(G, str(fam_dir / 'pedigree.nx'))
            prof.to_csv(fam_dir / 'profiles.txt', sep='\t', index=False, na_rep='NA')
            families.append(dict(name=name, seed=fs, G=G, prof=prof, work=work, dir=fam_dir,
                                 founders=founders))
            done += 1

        # 2) founder genome pools, shared by all families but never reused between them
        assignments = {}
        pools = []
        if cfg.genotypes:
            progress(done / steps, 'Preparing founder genomes')
            total = sum(len(f['founders']) for f in families)
            pools = _prepare_founder_pools(cfg, total, cfg.seed, tmp_root, _log)
            common = set(_samples(pools[0][1]))
            for _, p in pools[1:]:
                common &= set(_samples(p))
            common = sorted(common)
            biggest = max(len(f['founders']) for f in families)
            if len(common) < biggest:
                raise PipelineError(
                    f'Your VCF has {len(common)} samples, but a family needs {biggest} founders. '
                    'Use a VCF with more people, or choose a smaller family size.')
            order = list(master.permutation(common))
            if len(order) >= total:
                pos = 0
                for f in families:
                    n = len(f['founders'])
                    assignments[f['name']] = order[pos:pos + n]
                    pos += n
            else:
                _log(f'Note: only {len(order)} founder genomes for {total} founders in total, '
                     'so some founder genomes are reused across (not within) families.')
                for f in families:
                    assignments[f['name']] = list(master.choice(order, len(f['founders']), replace=False))
            done += 1

        # 3) per-family outputs
        summary, all_rel, all_ped = [], [], []
        for f in families:
            G, prof, fam_dir, work, name = f['G'], f['prof'], f['dir'], f['work'], f['name']
            row = dict(family=name, people=G.number_of_nodes(), founders=len(f['founders']),
                       generations=max(generation_layers(G, prof).values()) + 1,
                       males=int((prof['Sex'] == 'male').sum()), females=int((prof['Sex'] == 'female').sum()))
            if cfg.genotypes:
                progress(done / steps, f'Simulating DNA for {name}')
                row['variants'] = _simulate_genomes(G, fam_dir, work, pools, assignments[name], cfg,
                                                    f['seed'] % 1_000_000, _log)
                (fam_dir / 'founder_genomes.txt').write_text(
                    'pedigree_founder_id\tvcf_sample_id\n' + ''.join(
                        f'{fid}\t{s}\n' for fid, s in zip(sorted(f['founders'], key=int), assignments[name])))
                letter_genotypes(fam_dir / 'genotypes.vcf.gz', f['seed']).to_csv(
                    fam_dir / 'genotypes_letters.tsv.gz', sep='\t', index=False)
                _log(f'{name}: {row["variants"]:,} variants across {len(pools)} region(s)')
                done += 1
            if cfg.relationships:
                rel = _relationships(G, name, work, _log)
                rel.to_csv(fam_dir / 'relationships.csv', index=False)
                all_rel.append(rel)
                row['relative_pairs'] = len(rel)
            if cfg.ped:
                all_ped.append(_write_ped(G, prof, name, fam_dir / 'pedigree.ped'))
            if cfg.picture:
                draw_pedigree(G, prof, fam_dir / 'pedigree.png',
                              f'{name}  ·  {G.number_of_nodes()} people')
            summary.append(row)

        pd.DataFrame(summary).to_csv(run_dir / 'summary.csv', index=False)
        if all_rel:
            pd.concat(all_rel).to_csv(run_dir / 'all_relationships.csv', index=False)
        if all_ped:
            pd.concat(all_ped).to_csv(run_dir / 'all_families.ped', sep=' ', index=False, header=False)

        import json_export
        json_export.write_dataset_json(run_dir)

        _log(f'Finished in {time.time() - started:.1f}s. Saved to {run_dir}')
        progress(1.0, 'Done')
        return run_dir
    except Exception as e:
        _log(f'ERROR: {e}')
        (run_dir / 'FAILED.txt').write_text(str(e))
        raise
    finally:
        (run_dir / 'log.txt').write_text('\n'.join(log_lines) + '\n')
        shutil.rmtree(tmp_root, ignore_errors=True)
