"""
Beginner-friendly web UI for generating pedigree datasets with py_ped_sim.

Start it with:  ./run_ui.sh   (or: streamlit run ui/app.py inside the ped_sim env)
"""

import datetime as dt
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402
import problems as Q  # noqa: E402

st.set_page_config(page_title='Pedigree Dataset Generator', page_icon='🧬', layout='wide')

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  .step { font-size: 0.8rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
          color: #6b7280; margin-bottom: -0.6rem; }
  .muted { color: #6b7280; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


def step(n, title, subtitle=''):
    st.markdown(f'<div class="step">Step {n}</div>', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.markdown(f'<div class="muted">{subtitle}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# results viewer (used after a run and in the "My datasets" tab)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _letter_table(fam_dir: str):
    fam_dir = Path(fam_dir)
    if (fam_dir / 'genotypes_letters.tsv.gz').exists():
        return pd.read_csv(fam_dir / 'genotypes_letters.tsv.gz', sep='\t', dtype=str)
    if (fam_dir / 'genotypes.vcf.gz').exists():
        return P.letter_genotypes(fam_dir / 'genotypes.vcf.gz')
    return None


@st.cache_data(show_spinner='Drawing the family tree…')
def _pedigree_with_genotypes(fam_dir: str, variant: int) -> bytes:
    fam_dir = Path(fam_dir)
    G = nx.read_edgelist(str(fam_dir / 'pedigree.nx'), create_using=nx.DiGraph())
    prof = pd.read_csv(fam_dir / 'profiles.txt', sep='\t', dtype={'ID': str})
    row = _letter_table(str(fam_dir)).iloc[variant]
    buf = io.BytesIO()
    gts = {k: v if isinstance(v, str) and v != 'NA' else '?' for k, v in row.iloc[3:].items()}
    P.draw_pedigree(G, prof, buf, f'{fam_dir.name}  ·  gene {row["GENE"]} '
                    f'(chr {row["CHROM"]}, position {row["POS"]})', genotypes=gts,
                    highlight={k for k, v in gts.items() if v == '?'})
    return buf.getvalue()


def _zip_bytes(folder: Path, skip: tuple = ()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in folder.rglob('*'):
            if f.is_file() and not any(part in skip for part in f.relative_to(folder).parts):
                z.write(f, Path(folder.name) / f.relative_to(folder))
    return buf.getvalue()


def show_run(run_dir: Path, key: str):
    summary_path = run_dir / 'summary.csv'
    if (run_dir / 'FAILED.txt').exists() or not summary_path.exists():
        st.error('This run did not finish.')
        if (run_dir / 'FAILED.txt').exists():
            st.code((run_dir / 'FAILED.txt').read_text())
        if (run_dir / 'log.txt').exists():
            with st.expander('Log'):
                st.code((run_dir / 'log.txt').read_text())
        return

    summary = pd.read_csv(summary_path)
    c = st.columns(4)
    c[0].metric('Families', len(summary))
    c[1].metric('People in total', int(summary['people'].sum()))
    c[2].metric('People per family', f"{summary['people'].min()}–{summary['people'].max()}")
    if 'variants' in summary:
        c[3].metric('Variants per family (avg)', f"{int(summary['variants'].mean()):,}")
    else:
        c[3].metric('Genotypes', 'not simulated')

    st.markdown('**Saved in:**')
    st.code(str(run_dir), language=None)

    with st.expander('All families at a glance', expanded=len(summary) > 1):
        st.dataframe(summary, hide_index=True, use_container_width=True)

    fams = summary['family'].tolist()
    fam = st.selectbox('Look at a family', fams, key=f'{key}_fam')
    fam_dir = run_dir / fam
    row = summary[summary['family'] == fam].iloc[0]

    left, right = st.columns([3, 2], gap='large')
    with left:
        caption = 'Squares = male, circles = female. Lines go from parents (above) to children (below).'
        gts = _letter_table(str(fam_dir))
        choice = None
        if gts is not None and len(gts):
            labels = ['None'] + [f'{g}  ·  chr {c}, position {p}'
                                 for g, c, p in zip(gts['GENE'], gts['CHROM'], gts['POS'])]
            choice = st.selectbox('Show genotypes on the tree for variant', range(len(labels)),
                                  format_func=labels.__getitem__, key=f'{key}_{fam}_var',
                                  help='Writes each person\'s genotype for this variant under them: '
                                       'AA = both dominant, Aa = one of each, aa = both recessive.')
        if choice:
            st.image(_pedigree_with_genotypes(str(fam_dir), choice - 1),
                     caption=caption + ' Under each person: their genotype for this variant.')
        elif (fam_dir / 'pedigree.png').exists():
            st.image(str(fam_dir / 'pedigree.png'), caption=caption)
        else:
            st.info('No picture was made for this run.')
    with right:
        st.markdown(f"**{fam}** has **{row['people']} people** over **{row['generations']} generations**, "
                    f"of whom **{row['founders']}** are founders (people with no parents in the tree).")
        files = sorted(p.name for p in fam_dir.iterdir())
        st.markdown('Files for this family:')
        st.markdown('\n'.join(f'- `{f}`' for f in files))

    tabs = st.tabs(['👪 People', '🔗 Relationships', '🧬 Genotypes'])
    with tabs[0]:
        prof = pd.read_csv(fam_dir / 'profiles.txt', sep='\t', dtype=str)
        st.caption('Gen = the birth-cohort year used to simulate that generation (NA = married-in founder).')
        st.dataframe(prof, hide_index=True, use_container_width=True, height=300)
    with tabs[1]:
        rel_path = fam_dir / 'relationships.csv'
        if rel_path.exists():
            rel = pd.read_csv(rel_path, dtype=str)
            counts = rel['Relationship'].value_counts().rename_axis('Relationship').reset_index(name='Pairs')
            a, b = st.columns([2, 3], gap='large')
            a.dataframe(counts, hide_index=True, use_container_width=True, height=320)
            b.bar_chart(counts.set_index('Relationship'), horizontal=True, height=320)
            st.caption('MC = number of meioses (parent→child steps) between the two people · '
                       'GD = generation gap · RT = direct line, full or half relationship · '
                       'RC = the original py_ped_sim label · Relationship = standard name.')
            st.dataframe(rel, hide_index=True, use_container_width=True, height=300)
        else:
            st.info('Relationship table was not requested for this run.')
    with tabs[2]:
        vcf = fam_dir / 'genotypes.vcf.gz'
        letters = fam_dir / 'genotypes_letters.tsv.gz'
        if vcf.exists():
            st.caption('First variants. Each variant is given a random letter (GENE). Each cell is one '
                       'person\'s genotype: capital = dominant allele, small = recessive allele, so '
                       'AA = both dominant, Aa = one of each, aa = both recessive.')
            if letters.exists():
                gt = pd.read_csv(letters, sep='\t', dtype=str, nrows=15)
            else:
                gt = P.letter_genotypes(vcf, limit=15)
            st.dataframe(gt, hide_index=True, use_container_width=True)
        else:
            st.info('Genotypes were not simulated for this run.')

    st.divider()
    d1, d2, _ = st.columns([1, 1, 2])
    if d1.button('📦 Prepare ZIP download', key=f'{key}_zipbtn'):
        st.session_state[f'{key}_zip'] = _zip_bytes(run_dir)
    if st.session_state.get(f'{key}_zip'):
        d2.download_button('⬇️ Download ZIP', st.session_state[f'{key}_zip'],
                           file_name=f'{run_dir.name}.zip', mime='application/zip', key=f'{key}_dl')


# ----------------------------------------------------------------------------
# pedigree problems: remove genotypes, then list every genotype that fits
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _family(fam_dir: str):
    fam_dir = Path(fam_dir)
    G = Q.read_pedigree(fam_dir)
    prof = pd.read_csv(fam_dir / 'profiles.txt', sep='\t', dtype=str)
    return G, prof


@st.cache_data(show_spinner='Drawing the family tree…')
def _masked_tree(fam_dir: str, masked: tuple) -> bytes:
    G, prof = _family(fam_dir)
    buf = io.BytesIO()
    P.draw_pedigree(G, prof, buf, f'{Path(fam_dir).name}  ·  {len(masked)} people with genotypes removed',
                    highlight=set(masked))
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _read_tsv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


@st.cache_data(show_spinner='Drawing the family tree…')
def _solution_tree(fam_dir: str, sol_path: str, variant: int, masked: tuple, mtime: float) -> bytes:
    G, prof = _family(fam_dir)
    known = _letter_table(fam_dir).iloc[variant]
    sol = _read_tsv(sol_path, mtime).iloc[variant]
    gts = {k: v for k, v in known.iloc[3:].items() if isinstance(v, str) and v != 'NA'}
    for p in masked:
        if sol[p] != '-':
            gts[p] = sol[p]
    buf = io.BytesIO()
    P.draw_pedigree(G, prof, buf, f'{Path(fam_dir).name}  ·  gene {sol["GENE"]} (chr {sol["CHROM"]}, '
                    f'position {sol["POS"]})  ·  {sol["COMBINATIONS"]} valid combination(s)',
                    genotypes=gts, highlight={p for p in masked if sol[p] != '-'})
    return buf.getvalue()


def problem_maker():
    sources = Q.source_datasets()
    if not sources:
        st.info('First make a dataset **with genotypes** in the **New dataset** tab.')
        return

    step(1, 'Pick the original dataset', 'Only datasets with genotypes are listed. The original is never changed.')
    by_name = {r.name: r for r in sources}
    src_name = st.selectbox('Original dataset', list(by_name), key='pm_src', help='Newest first.')
    src = by_name[src_name]
    summary = pd.read_csv(src / 'summary.csv')
    families = summary['family'].tolist()

    st.write('')
    step(2, 'Choose whose genotypes to remove',
         'Pick a family, then the people. Your choices are remembered for each family, so you can '
         'remove people in several families before saving.')
    fam = st.selectbox('Family', families, key=f'pm_fam_{src_name}')
    fam_dir = src / fam
    G, prof = _family(str(fam_dir))
    sex = dict(zip(prof['ID'], prof['Sex']))
    layer = P.generation_layers(G, prof)
    ids = sorted(G.nodes, key=int)
    founders = {n for n in ids if G.in_degree(n) == 0}
    # Widget state is dropped when a widget is not shown, so choices live in their own store.
    store = st.session_state.setdefault('pm_masks', {}).setdefault(src_name, {})
    sel_key = f'pm_sel_{src_name}_{fam}'
    if sel_key not in st.session_state:
        st.session_state[sel_key] = store.get(fam, [])

    def set_selection(v):
        store[fam] = sorted(set(v), key=int)
        st.session_state[sel_key] = store[fam]

    def add_generation(g):
        set_selection(set(st.session_state.get(sel_key, [])) | {n for n in ids if layer[n] == g})

    left, right = st.columns([3, 2], gap='large')
    with right:
        selected = st.multiselect(
            'People whose genotypes will be removed', ids, key=sel_key,
            on_change=lambda: store.update({fam: list(st.session_state[sel_key])}),
            format_func=lambda n: f'{n} · {sex.get(n, "?")} · gen {layer[n] + 1}' + (' · founder' if n in founders else ''))
        g1, g2 = st.columns([2, 1])
        gen = g1.selectbox('Quick pick: a whole generation', range(max(layer.values()) + 1),
                           format_func=lambda g: f'Generation {g + 1} ({sum(layer[n] == g for n in ids)} people)',
                           key=f'pm_gen_{src_name}_{fam}')
        g2.write('')
        g2.button('Add them', on_click=add_generation, args=(gen,), key=f'pm_addgen_{src_name}_{fam}',
                  use_container_width=True)
        st.button('Clear this family', on_click=set_selection, args=([],),
                  key=f'pm_clear_{src_name}_{fam}')
        st.markdown(f'<div class="muted">{len(selected)} of {len(ids)} people chosen in {fam}.</div>',
                    unsafe_allow_html=True)
    with left:
        st.image(_masked_tree(str(fam_dir), tuple(selected)),
                 caption='Grey = genotypes will be removed. Squares = male, circles = female.')

    masks = {f: sorted(store.get(f, []), key=int) for f in families if store.get(f)}

    st.write('')
    step(3, 'Which variants?', 'Usually the whole genome. You can also remove only part of it.')
    scope_choice = st.radio('Remove their genotypes at', ['Every variant (whole genome)', 'Only some variants'],
                            horizontal=True, key='pm_scope', label_visibility='collapsed')
    scope = None
    if scope_choice == 'Only some variants':
        chroms = sorted(_letter_table(str(src / families[0]))['CHROM'].astype(str).unique().tolist())
        c = st.columns([2, 1, 1])
        chosen = c[0].multiselect('Chromosomes', chroms, default=chroms, key='pm_chroms')
        start = c[1].number_input('From position (bp, 0 = start)', 0, 10**10, 0, key='pm_start')
        end = c[2].number_input('To position (bp, 0 = end)', 0, 10**10, 0, key='pm_end')
        scope = dict(chromosomes=chosen, start=int(start) or None, end=int(end) or None)

    st.write('')
    step(4, 'Save the problem')
    if masks:
        st.dataframe(pd.DataFrame([dict(family=f, removed=len(v), people=', '.join(v)) for f, v in masks.items()]),
                     hide_index=True, use_container_width=True)
    name = st.text_input('Name of the new dataset', value=f'{src_name}__problem', key=f'pm_name_{src_name}',
                         help='Saved as a new folder in outputs/. If the name is taken, _2, _3, … is added.')
    problems = []
    if not masks:
        problems.append('Choose at least one person.')
    if scope is not None and not scope['chromosomes']:
        problems.append('Choose at least one chromosome.')
    if scope is not None and scope['start'] and scope['end'] and scope['start'] > scope['end']:
        problems.append('"From position" is after "To position".')
    for p in problems:
        st.warning(p)
    if st.button('💾 Save problem dataset', type='primary', disabled=bool(problems), use_container_width=True):
        try:
            with st.spinner('Removing genotypes…'):
                out = Q.make_problem(src, masks, name, scope)
            st.session_state['last_problem'] = str(out)
            st.success(f'Saved as **outputs/{out.name}**. Switch to **Solve a problem** to work out the answers.')
        except Exception as e:
            st.error(str(e))


def show_solution(run_dir: Path, problem: dict):
    sol_dir = run_dir / Q.SOLUTION_DIR
    summ = pd.read_csv(sol_dir / 'solution_summary.csv', dtype={'ID': str})
    meta = json.loads((sol_dir / 'solution.json').read_text())
    tot = summ['masked_variants'].sum()
    c = st.columns(4)
    c[0].metric('People to solve', len(summ))
    c[1].metric('Exactly one answer', f"{100 * summ['one_possible'].sum() / max(tot, 1):.1f}%",
                help='Share of (person, variant) cells where only one genotype fits.')
    c[2].metric('Two possible', f"{100 * summ['two_possible'].sum() / max(tot, 1):.1f}%")
    c[3].metric('Any genotype possible', f"{100 * summ['three_possible'].sum() / max(tot, 1):.1f}%")
    if 'true_in_possible_pct' in summ:
        worst = summ['true_in_possible_pct'].min()
        if worst == 100:
            st.success('Answer check: the true genotype (from the original dataset) is among the possible ones '
                       'for every person at every variant.')
        else:
            st.warning('Answer check: at some variants the true genotype is not among the possible ones. This '
                       'happens when a new mutation in the simulation broke Mendel\'s rules there (see the '
                       'answer check table below).')
    else:
        st.info('The original dataset was not found, so the answers could not be checked against the truth.')
    st.caption(f"Solved {meta['solved']} · {meta['method']}")
    st.dataframe(summ, hide_index=True, use_container_width=True)

    fams = list(problem['masked'])
    fam = st.selectbox('Look at a family', fams, key=f'ps_fam_{run_dir.name}')
    masked = tuple(problem['masked'][fam])
    sol_path = sol_dir / f'{fam}_possible_genotypes.tsv.gz'
    mtime = sol_path.stat().st_mtime
    sol = _read_tsv(str(sol_path), mtime)

    only_amb = st.checkbox('Only list variants where someone has more than one possible genotype', value=True,
                           key=f'ps_amb_{run_dir.name}')
    rows = range(len(sol))
    if only_amb:
        amb = sol[list(masked)].apply(lambda col: col.str.contains('/')).any(axis=1)
        rows = [i for i in rows if amb.iat[i]]
    if not rows:
        st.info('Every removed genotype in this family has exactly one possible answer.')
        return
    v = st.selectbox('Variant', rows, key=f'ps_var_{run_dir.name}_{fam}_{only_amb}',
                     format_func=lambda i: f'{sol["GENE"].iat[i]}  ·  chr {sol["CHROM"].iat[i]}, position '
                                           f'{sol["POS"].iat[i]}  ·  {sol["COMBINATIONS"].iat[i]} combination(s)')
    st.image(_solution_tree(str(run_dir / fam), str(sol_path), v, masked, mtime),
             caption='Grey = genotype was removed; the label lists every genotype that person could have '
                     '(e.g. Aa/aa = Aa or aa). Others show their known genotype.')

    tabs = st.tabs(['🧩 Possible genotypes', '🔢 Combinations', '✅ Answer check'])
    with tabs[0]:
        st.caption('One row per variant. COMBINATIONS = number of ways to fill in all removed genotypes in the '
                   'family at once. "-" = known at this variant. CONFLICT = the known relatives break Mendel\'s rules.')
        st.dataframe(sol.iloc[list(rows)] if only_amb else sol, hide_index=True, use_container_width=True, height=320)
    with tabs[1]:
        comb_path = sol_dir / f'{fam}_combinations.tsv.gz'
        if comb_path.exists():
            comb = _read_tsv(str(comb_path), comb_path.stat().st_mtime)
            here = comb[(comb['CHROM'] == sol['CHROM'].iat[v]) & (comb['POS'] == sol['POS'].iat[v])]
            st.caption(f'Every valid way to fill in the removed genotypes at the selected variant '
                       f'(listing stops at {meta["max_combinations_per_variant"]:,} per variant).')
            st.dataframe(here, hide_index=True, use_container_width=True, height=320)
        else:
            st.info('Combinations were not listed. Tick **Also list every combination** and solve again.')
    with tabs[2]:
        chk_path = sol_dir / f'{fam}_answer_check.tsv.gz'
        if chk_path.exists():
            chk = _read_tsv(str(chk_path), chk_path.stat().st_mtime)
            miss = chk[chk['TRUE_IN_POSSIBLE'] != 'True']
            st.caption('TRUE = genotype in the original dataset · POSSIBLE = what the solver allows.')
            if len(miss):
                st.markdown(f'**{len(miss)} cells** where the true genotype is not among the possible ones:')
                st.dataframe(miss, hide_index=True, use_container_width=True, height=240)
            st.dataframe(chk, hide_index=True, use_container_width=True, height=320)
        else:
            st.info('No answer check (the original dataset was not found).')

    st.divider()
    d = st.columns(4)
    if d[0].button('📦 Prepare ZIPs', key=f'ps_zip_{run_dir.name}'):
        st.session_state[f'ps_zip_{run_dir.name}'] = (_zip_bytes(run_dir, skip=(Q.SOLUTION_DIR,)), _zip_bytes(run_dir))
    if st.session_state.get(f'ps_zip_{run_dir.name}'):
        only, full = st.session_state[f'ps_zip_{run_dir.name}']
        d[1].download_button('⬇️ Problem only', only, file_name=f'{run_dir.name}.zip', mime='application/zip',
                             help='Without the solution folder, ready to hand out.', key=f'ps_dl1_{run_dir.name}')
        d[2].download_button('⬇️ Problem + solution', full, file_name=f'{run_dir.name}_with_solution.zip',
                             mime='application/zip', key=f'ps_dl2_{run_dir.name}')


def problem_solver():
    probs = Q.problem_datasets()
    if not probs:
        st.info('No problem datasets yet. Make one with **Make a problem**.')
        return
    by_name = {r.name: r for r in probs}
    names = list(by_name)
    last = Path(st.session_state.get('last_problem', '')).name
    run_name = st.selectbox('Problem dataset', names, index=names.index(last) if last in names else 0,
                            key='ps_run', help='Newest first.')
    run_dir = by_name[run_name]
    problem = Q.read_problem(run_dir)
    scope = problem.get('scope')
    where = 'every variant' if not scope else (
        f"chromosome(s) {', '.join(map(str, scope['chromosomes']))}"
        + (f", from {scope['start']:,} bp" if scope.get('start') else '')
        + (f", to {scope['end']:,} bp" if scope.get('end') else ''))
    st.markdown(f"Made from **{problem['source']}**. Genotypes removed at {where} for: "
                + '; '.join(f"**{f}**: {', '.join(v)}" for f, v in problem['masked'].items()))

    c = st.columns([2, 1, 1])
    want_comb = c[0].checkbox('Also list every combination', key=f'ps_comb_{run_name}',
                              help='Every valid way to fill in all removed genotypes of a family together, per '
                                   'variant. Can be very many, so the list is cut off at the limit.')
    max_comb = c[1].number_input('Limit per variant', 1, 1_000_000, 1000, disabled=not want_comb,
                                 key=f'ps_max_{run_name}')
    solved = (run_dir / Q.SOLUTION_DIR / 'solution_summary.csv').exists()
    c[2].write('')
    if c[2].button('🧠 Solve again' if solved else '🧠 Solve', type='primary', use_container_width=True,
                   key=f'ps_solve_{run_name}'):
        with st.status('Working out every possible genotype…', expanded=True) as status:
            bar = st.progress(0.0, text='Starting')
            try:
                Q.solve_problem(run_dir, want_comb, int(max_comb),
                                progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                status.update(label='Solved', state='complete', expanded=False)
                solved = True
            except Exception as e:
                status.update(label='Something went wrong', state='error', expanded=True)
                st.error(str(e))
    if solved:
        st.divider()
        show_solution(run_dir, problem)


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------

st.title('🧬 Pedigree Dataset Generator')
st.markdown('<div class="muted">Make simulated family trees, optionally with DNA for everyone in them. '
            'Fill in the steps from top to bottom and press <b>Generate</b>. '
            f'Everything is saved in the <code>outputs/</code> folder.</div>', unsafe_allow_html=True)
st.write('')

tab_new, tab_runs, tab_prob, tab_help = st.tabs(['➕ New dataset', '📁 My datasets', '🧩 Pedigree problems',
                                                 '❓ Help'])

with tab_new:
    years_all = P.available_years()

    # Step 1 ------------------------------------------------------------------
    step(1, 'Name your dataset')
    c = st.columns([2, 1, 1])
    run_name = c[0].text_input('Dataset name', value=f'dataset_{dt.datetime.now():%Y%m%d_%H%M}',
                               help='Used as the folder name inside outputs/. Spaces become underscores.')
    n_families = c[1].number_input('How many families?', 1, 500, 5,
                                   help='Each family is a separate, unrelated pedigree.')
    seed = c[2].number_input('Random seed', 0, 10**9, 42,
                             help='Same seed + same settings = the same dataset again.')

    # Step 2 ------------------------------------------------------------------
    st.write('')
    step(2, 'Choose the family size', 'Bigger families have more generations. Sizes vary randomly, '
         'so you can also set a range of people to keep.')
    preset_names = list(P.PRESETS) + ['Custom']
    preset = st.radio('Family size', preset_names, index=1, horizontal=True, label_visibility='collapsed')
    if preset == 'Custom':
        years = st.multiselect(
            'Birth-cohort years: each year adds one generation of children below the founding couple', years_all,
            default=[1900, 1950, 2000],
            help='The year picks how many children each couple tends to have (US census data): '
                 'about 3.3 in 1850, 2.5 in 1930 and 1.8 today. Earlier years = bigger families.')
        years = sorted(years)
        size_default = (2, 500)
    else:
        years = P.PRESETS[preset]['years']
        size_default = P.PRESETS[preset]['size']
        st.info(f"**{preset}:** {P.PRESETS[preset]['blurb']}  ·  generations from: "
                f"{', '.join(map(str, years))}")

    min_size, max_size = st.slider('Keep only families with this many people', 1, 1000, size_default,
                                   key=f'size_{preset}',
                                   help='Families outside this range are thrown away and re-drawn.')

    with st.expander('More family options'):
        broaden = st.checkbox(
            'Add in-law families (wider trees)',
            help='Also simulates the families of people who married in, and joins them on. '
                 'Makes much larger pedigrees. Note: these extra families do not follow the seed.')
        half_sibs = st.checkbox('Include half-siblings',
                                help='Some children of the family\'s daughters get a different father '
                                     '(from outside the family), making them half-siblings of their brothers and sisters.')
        hs_prob = st.slider('Chance that such a child has a different father (%)', 1, 50, 10,
                            disabled=not half_sibs) / 100
        max_attempts = st.number_input('Max re-draws per family', 10, 2000, 200,
                                       help='Give up if no family fits the size range after this many tries.')

    # Step 3 ------------------------------------------------------------------
    st.write('')
    step(3, 'Genetic data', 'Simulate DNA (genotypes) for every person, inherited down the family tree.')
    genotypes = st.toggle('Simulate genotypes (DNA)', value=True)
    uploads, regions_df = [], None
    mutation_rate, ne = '1e-8', 10000
    source = 'simulate'
    if genotypes:
        source_label = st.radio('Where should the founders\' DNA come from?',
                                ['Simulate it for me (recommended)', 'Use my own VCF file(s)'],
                                horizontal=True)
        source = 'simulate' if source_label.startswith('Simulate') else 'upload'
        if source == 'simulate':
            st.markdown('Each row is one chromosome or region. Add rows for more chromosomes. '
                        'Longer regions = more variants (about 1,500 variants per 1,000 kb).')
            default_regions = pd.DataFrame([{'Chromosome': '1', 'Length (kb)': 1000, 'Recombination rate': '1e-8'}])
            regions_df = st.data_editor(
                default_regions, num_rows='dynamic', use_container_width=True, hide_index=True,
                column_config={
                    'Chromosome': st.column_config.TextColumn(help='Name used in the VCF, e.g. 1, 2, X'),
                    'Length (kb)': st.column_config.NumberColumn(min_value=10, max_value=50_000, step=10,
                                                                 help='1,000 kb = 1 Mb'),
                    'Recombination rate': st.column_config.SelectboxColumn(
                        options=P.RATES, help='Per base per generation. 1e-8 is typical for humans.'),
                })
            with st.expander('Advanced DNA settings'):
                a, b = st.columns(2)
                mutation_rate = a.selectbox('Mutation rate (per base per generation)', P.RATES, index=1)
                ne = b.number_input('Population size the founders come from', 1000, 1_000_000, 10000, step=1000,
                                    help='Bigger = more genetic diversity among founders.')
        else:
            st.markdown('Upload one VCF per chromosome. Each needs at least as many samples as founders in a '
                        'family; ideally enough for every founder in the whole dataset.')
            uploads = st.file_uploader('VCF files', type=['vcf', 'gz'], accept_multiple_files=True)
            with st.expander('Advanced DNA settings'):
                mutation_rate = st.selectbox('Mutation rate (per base per generation)', P.RATES, index=1)

    # Step 4 ------------------------------------------------------------------
    st.write('')
    step(4, 'Extra outputs')
    c = st.columns(3)
    want_rel = c[0].checkbox('Relationship table', value=True,
                             help='Every pair of relatives with their relationship (cousin, aunt, ...).')
    want_ped = c[1].checkbox('.ped file', value=True, help='Standard pedigree format used by PLINK and others.')
    want_pic = c[2].checkbox('Family picture', value=True)

    # Generate ------------------------------------------------------------------
    st.write('')
    problems = []
    if not years:
        problems.append('Pick at least one generation year.')
    if min_size > max_size:
        problems.append('The minimum family size is bigger than the maximum.')
    regions = []
    if genotypes and source == 'simulate':
        rdf = regions_df.dropna(how='all') if regions_df is not None else pd.DataFrame()
        if rdf.empty:
            problems.append('Add at least one chromosome/region.')
        names = rdf['Chromosome'].astype(str).str.strip().tolist() if not rdf.empty else []
        if len(set(names)) != len(names) or any(n in ('', 'None', 'nan') for n in names):
            problems.append('Each chromosome needs a different, non-empty name.')
        for _, r in rdf.iterrows():
            if pd.isna(r['Length (kb)']) or r['Length (kb)'] <= 0:
                problems.append('Every region needs a length.')
                break
            regions.append(P.Region(str(r['Chromosome']).strip(), int(r['Length (kb)'] * 1000),
                                    r['Recombination rate'] if r['Recombination rate'] in P.RATES else '1e-8'))
    if genotypes and source == 'upload' and not uploads:
        problems.append('Upload at least one VCF file (or switch to "Simulate it for me").')

    for p in problems:
        st.warning(p)

    if st.button('🚀 Generate dataset', type='primary', disabled=bool(problems), use_container_width=True):
        upload_dir = Path(tempfile.mkdtemp(prefix='pedsim_upload_'))
        upload_paths = []
        for u in uploads or []:
            dest = upload_dir / Path(u.name).name.replace(' ', '_')
            dest.write_bytes(u.getbuffer())
            upload_paths.append(str(dest))

        cfg = P.RunConfig(
            run_name=run_name, n_families=int(n_families), seed=int(seed), years=list(years),
            min_size=int(min_size), max_size=int(max_size), max_attempts=int(max_attempts),
            broaden=broaden, half_sibs=half_sibs, half_sib_prob=hs_prob,
            genotypes=genotypes, founder_source=source, regions=regions or [P.Region()],
            uploaded_vcfs=upload_paths, ne=int(ne), mutation_rate=mutation_rate,
            relationships=want_rel, ped=want_ped, picture=want_pic)

        with st.status('Generating your dataset...', expanded=True) as status:
            bar = st.progress(0.0, text='Starting')
            log_box = st.empty()
            lines = []

            def log(msg):
                lines.append(msg)
                log_box.code('\n'.join(lines[-12:]), language=None)

            try:
                run_dir = P.run_pipeline(cfg, log=log, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                status.update(label=f'Done! Saved to outputs/{run_dir.name}', state='complete', expanded=False)
                st.session_state['last_run'] = str(run_dir)
            except Exception as e:
                status.update(label='Something went wrong', state='error', expanded=True)
                st.error(str(e))
            finally:
                shutil.rmtree(upload_dir, ignore_errors=True)

    if st.session_state.get('last_run') and Path(st.session_state['last_run']).exists():
        st.divider()
        st.success(f"Your dataset **{Path(st.session_state['last_run']).name}** is ready.")
        show_run(Path(st.session_state['last_run']), key='new')


with tab_runs:
    runs = P.list_runs()
    if not runs:
        st.info('No datasets yet. Make one in the **New dataset** tab.')
    else:
        labels = {r.name: r for r in runs}
        chosen = st.selectbox('Dataset', list(labels), help='Newest first.')
        run_dir = labels[chosen]
        show_run(run_dir, key='browse')
        with st.expander('🗑️ Delete this dataset'):
            sure = st.checkbox(f'Yes, permanently delete outputs/{chosen}', key=f'del_{chosen}')
            if st.button('Delete', disabled=not sure, key=f'delbtn_{chosen}'):
                shutil.rmtree(run_dir)
                if st.session_state.get('last_run') == str(run_dir):
                    st.session_state.pop('last_run')
                st.rerun()


with tab_prob:
    st.markdown('<div class="muted">Turn a dataset into a puzzle: remove the genotypes of people you choose, '
                'then let the app list <b>every genotype each of them could have had</b>, judged only from '
                'their relatives\' known genotypes and Mendel\'s rules (one variant at a time).</div>',
                unsafe_allow_html=True)
    st.write('')
    mode = st.radio('Mode', ['✂️ Make a problem', '🧠 Solve a problem'], horizontal=True,
                    label_visibility='collapsed', key='prob_mode')
    st.write('')
    if mode.startswith('✂️'):
        problem_maker()
    else:
        problem_solver()


with tab_help:
    st.markdown("""
### How it works
1. **A family tree is simulated.** It starts with one founder couple. Each couple has a random
   number of children, drawn from real US-census family sizes for the year of that generation.
   Children then marry people from outside the family (new *founders*), and so on.
2. **Founder DNA is created** with a population-genetics simulator (msprime), or taken from your own VCF.
   Every founder gets a different genome, and no genome is used in two families.
3. **DNA is passed down** the tree with SLiM, including recombination and new mutations, so every
   child really inherits half of each parent's DNA.
4. **Relationships are listed** for every pair of relatives.

### Words you will see
| Word | Meaning |
|---|---|
| **Founder** | Someone at the top of the tree or who married in. Their parents are not in the tree. |
| **Generation** | One level of the tree. A *Small* family has 4: the founding couple, children, grandchildren and great-grandchildren. |
| **Half-siblings** | Children who share only one parent. |
| **Genotype / VCF** | The DNA letters at each variable position, for each person. A `.vcf.gz` opens with `bcftools`, `plink` or Python (`cyvcf2`, `scikit-allel`). |
| **.ped file** | A text table: family, person, father, mother, sex (1 = male, 2 = female), phenotype (-9 = unknown). |
| **MC / GD / RT** | Meiosis count (parent→child steps between two people), generation gap, and direct / full / half. |
| **Seed** | A number that makes the randomness repeatable. |

### Pedigree problems
In the **🧩 Pedigree problems** tab you remove the genotypes of chosen people from a dataset and save that
as a new dataset (named after the original, e.g. `mydata__problem`). **Solve** then lists, for each removed
person and each variant, every genotype that is possible given everyone else's known genotypes, for
example `Aa/aa`. People with no informative relatives can be anything (`AA/Aa/aa`). Answers are worked
out for the whole family at once, so removed relatives constrain each other correctly. Optionally it
also lists every valid *combination* for all removed people together. The answers are saved in a
`solution/` folder inside the problem dataset; download "Problem only" to hand out the puzzle without them.

### Where are my files?
In the `outputs/` folder inside the project, one folder per dataset. Each dataset has a `README.txt`
explaining every file, and the exact settings in `settings.json`.

### Tips
- For **machine-learning datasets**, make many families (for example 100) with the same settings. Each
  family is an independent sample.
- The family size changes a lot by chance. Use the **size range** slider to keep only families of the size you want.
- More chromosomes or longer regions give more variants but take longer. 1,000 kb is a good start.
""")
