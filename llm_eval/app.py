"""
app.py — the LLM Genotype Evaluation app (Streamlit)
=====================================================

Start it with  ./run_llm_eval.sh  from the project folder (opens http://localhost:8502).

The page has a sidebar and four tabs, which follow the order of an evaluation:

  sidebar        pick a problem dataset (made and solved in the Pedigree Dataset Generator app)
  1 · Prompt     see exactly what the LLM will be given, and the leak-check results
  2 · Answer     get an answer: ask Claude, ask an OpenRouter model (e.g. DeepSeek), or
                 paste/upload one from any LLM; shows live how many variants are answered
  3 · Results    coverage (which variants were answered), then validity, uniqueness and recovery
                 rates for one run, with every hypothesis
  4 · Compare    all runs of this problem side by side

The heavy lifting lives in the other files:
  data.py     finding/loading problems and answer keys
  prompt.py   building the prompt + leak check
  llm.py      calling Claude (Anthropic API)
  llm_openrouter.py  calling any OpenRouter model, e.g. DeepSeek V4 Flash
  scoring.py  turning a reply into hypotheses and scoring them
  runs.py     saving, re-scoring and comparing runs
"""

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd
import streamlit as st

import data as D
import llm as L
import llm_openrouter as LO
import prompt as PR
import runs as R
import scoring as S

st.set_page_config(page_title='LLM Genotype Evaluation', page_icon='🧪', layout='wide')

ONE_REQUEST = 'One request for the whole problem'
PER_FAMILY = 'One request per family'
PER_VARIANT = 'One request per variant'
MODES = [ONE_REQUEST, PER_FAMILY, PER_VARIANT]

METRIC_HELP = {
    'validity': 'Of every hypothesis the LLM suggested (repeats included), the % that are admissible '
                '— i.e. that genotype really is possible for that person at that variant.',
    'uniqueness': 'Distinct hypotheses ÷ all suggested hypotheses. 100% = it never repeated itself.',
    'recovery': 'Distinct admissible hypotheses found ÷ every admissible hypothesis in the answer key. '
                '100% = it found the complete answer.',
}


def fmt_pct(x) -> str:
    return '—' if x is None or (isinstance(x, float) and pd.isna(x)) else f'{x:.2f}%'


# ==============================================================================================
# Sidebar — choose the problem
# ==============================================================================================
# Only problem datasets are listed (folders with problem_dataset.json). The answer key must exist
# (the problem was "Solved" in the generator app) before an answer can be scored.

st.sidebar.title('🧪 LLM evaluation')
problems = D.list_problems()
if not problems:
    st.title('🧪 LLM Genotype Evaluation')
    st.info(f'No problem datasets found in `{D.OUTPUTS}`. Make one in the Pedigree Dataset Generator '
            '(🧩 Pedigree problems → Make a problem), solve it there, then come back.')
    st.stop()

names = [p.name for p in problems]
problem_dir = problems[names.index(st.sidebar.selectbox('Problem dataset', names, help='Newest first.'))]
problem = D.load_problem(problem_dir)
solution = D.load_solution(problem_dir)

n_people = sum(len(f['people']) for f in problem['families'])
n_variants = sum(len(f.get('variants', [])) for f in problem['families'])
st.sidebar.markdown(f"**{len(problem['families'])}** families · **{n_people}** people · "
                    f"**{n_variants:,}** variants")
if solution is None:
    st.sidebar.warning('Not solved yet: answers can be collected but not scored. '
                       'Run **Solve** for this problem in the generator app.')
else:
    st.sidebar.success('Answer key found.')
st.sidebar.caption(f'Datasets folder: `{D.OUTPUTS}`\n\nSaved runs: `{R.RUNS}`')

st.title('🧪 LLM Genotype Evaluation')
st.caption(f'Problem **{problem_dir.name}** · the LLM must list every genotype each person with a removed '
           'genotype could have, using only the pedigree, the known genotypes and Mendelian inheritance.')

tab_prompt, tab_answer, tab_results, tab_compare = st.tabs(
    ['1 · Prompt', '2 · Answer', '3 · Results', '4 · Compare runs'])


# ==============================================================================================
# Building the prompt(s)
# ==============================================================================================
# A problem can be sent as one request, or split into one request per family, or one per variant
# (smaller prompts and much shorter replies; every family and every variant is solved on its own
# anyway, so nothing is lost by splitting). `parts` maps a part name to its prompt:
#   ''                         the single request
#   'family_001'               one family (all its variants)
#   'family_001__1-331'        one variant of one family (file-name safe: ':' becomes '-')
# Only variants with at least one removed genotype get a request of their own.
TARGETS = PR.targets(problem)          # [(family, variant, removed genotypes), ...]


def make_parts(mode: str) -> dict:
    if mode == ONE_REQUEST:
        groups = {'': (None, None)}
    elif mode == PER_FAMILY:
        groups = {f['family']: ([f['family']], None) for f in problem['families']}
    else:
        groups = {f'{fam}__{vid.replace(":", "-")}': ([fam], [vid]) for fam, vid, _ in TARGETS}
    parts = {}
    for name, (fams, variants) in groups.items():
        system, user, view = PR.build_prompt(problem, fams, variants)
        check = PR.leak_check(view, PR.prompt_as_text(system, user), problem, solution) if solution else None
        cells = sum(g is None for f in view for v in f['variants'] for g in v['genotypes'].values())
        parts[name] = dict(system=system, user=user, view=view, check=check, cells=cells,
                           variants=sum(1 for f in view for v in f['variants']
                                        if any(g is None for g in v['genotypes'].values())),
                           family=fams[0] if fams and len(fams) == 1 else '')
    return parts


# ==============================================================================================
# Tab 1 — what the LLM sees
# ==============================================================================================

with tab_prompt:
    mode = st.radio('How to send the problem', MODES, horizontal=True, key='mode',
                    help='Splitting keeps every request and reply small. A model that has to write a '
                         'very long answer often stops after a few variants; one request per variant '
                         'avoids that. Pasting into a chat by hand works best with one request.')
    parts = make_parts(mode)
    n_cells = sum(n for *_, n in TARGETS)
    biggest = max(parts.values(), key=lambda p: p['cells'])
    st.caption(f'The answer has to cover **{len(TARGETS):,}** variants with **{n_cells:,}** removed genotypes. '
               f'{len(parts):,} request(s); the largest asks for {biggest["cells"]:,} genotypes in '
               f'{biggest["variants"]:,} variant(s).')

    # Leak check summary (one line per test; all parts must pass).
    st.subheader('Leak check')
    if solution is None:
        st.info('The leak check needs the answer key. Solve the problem first.')
    else:
        all_ok = all(p['check']['ok'] for p in parts.values())
        (st.success if all_ok else st.error)(
            'Passed: the LLM sees no part of the answer.' if all_ok else
            'FAILED: something from the answer would be shown. Nothing can be sent until this is fixed.')
        first = next(iter(parts.values()))['check']
        rows = []
        for name, ok, detail in first['checks']:
            ok_all = all(dict((c[0], c[1]) for c in p['check']['checks'])[name] for p in parts.values())
            rows.append({'Check': name, 'Result': '✅' if ok_all else '❌',
                         'Detail': detail if len(parts) == 1 else ('all parts pass' if ok_all else detail)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption('The LLM is also given no tools (no web, no code, no files), so this prompt is all it knows.')

    # The prompt itself.
    st.subheader('Prompt')
    total_chars = sum(len(p['system']) + len(p['user']) for p in parts.values())
    st.caption(f'{len(parts)} request(s) · {total_chars:,} characters in total '
               f'(roughly {total_chars // 4:,} tokens — an estimate).')
    with st.expander('Instructions given to the LLM', expanded=False):
        st.code(PR.SYSTEM_PROMPT + '\n\n' + PR.TASK_INSTRUCTIONS, language=None)
    pick = st.selectbox('Show the full prompt for', list(parts),
                        format_func=lambda p: p or 'the whole problem') if len(parts) > 1 else ''
    with st.expander('Full prompt text', expanded=False):
        txt = PR.prompt_as_text(parts[pick]['system'], parts[pick]['user'])
        st.code(txt[:20000] + ('\n… (truncated here for display; the download has everything)'
                               if len(txt) > 20000 else ''), language=None)
    single = make_parts(ONE_REQUEST)['']
    st.download_button('⬇️ Download prompt (whole problem, for any LLM)',
                       PR.prompt_as_text(single['system'], single['user']),
                       file_name=f'{problem_dir.name}_prompt.txt', mime='text/plain',
                       disabled=bool(solution and not single['check']['ok']))


# ==============================================================================================
# Tab 2 — getting an answer
# ==============================================================================================

with tab_answer:
    blocked = solution is not None and not all(p['check']['ok'] for p in parts.values())
    if blocked:
        # Not st.stop(): that would also hide the Results and Compare tabs.
        st.error('The leak check failed (tab 1), so nothing can be sent.')
    CLAUDE, OPENROUTER, MANUAL = ('Ask Claude (Anthropic API)', 'Ask a model on OpenRouter (e.g. DeepSeek)',
                                  'Paste or upload an answer from any LLM')
    how = st.radio('Where does the answer come from?', [CLAUDE, OPENROUTER, MANUAL], horizontal=True)
    label = st.text_input('Name for this run', value='', placeholder='e.g. deepseek-flash-high-try1',
                          help='Shown in results and comparisons. Letters, numbers, - and _.')

    workers = 1
    if how != MANUAL and len(parts) > 1:
        workers = int(st.number_input('Requests sent at the same time', 1, 16, 4,
                                      help='Requests run in parallel to save time. Lower it if the provider '
                                           'reports rate limits.'))

    def send_and_save(ask, model: str, info: dict, default_label: str):
        """Shared by both API providers: send every part (one request, or one per family / variant)
        with `ask(system, user, on_text)`, `workers` at a time, then save and score the run.

        While the requests run, the page shows live how many variants the replies have started to
        answer (read from the streamed text). If a request fails outright (bad key, no credits,
        network), no new requests are started; the replies that did arrive are still saved and the
        failed requests are recorded with the run. Nothing is saved if every request failed."""
        names = list(parts)
        live = {n: [] for n in names}              # streamed answer pieces per part (worker threads append)

        def collector(name):
            def on_text(piece):
                if piece is None:                    # the provider retried: start this part again
                    live[name].clear()
                else:
                    live[name].append(piece)
            return on_text

        def run_one(name):
            p = parts[name]
            return ask(p['system'], p['user'], collector(name))

        bar, status = st.progress(0.0, text='Starting'), st.empty()
        results, started = {}, time.time()

        def show_progress(running: int):
            seen = set()
            for n in names:
                seen |= S.variants_in_text(''.join(live[n]), parts[n]['family'])
            covered = len(seen & {(f, v) for f, v, _ in TARGETS})
            bar.progress(min(covered / max(len(TARGETS), 1), 1.0),
                         text=f'Variants answered so far: {covered:,} / {len(TARGETS):,}')
            status.caption(f'Waiting for {model} · requests: {len(results):,} / {len(names):,} done, '
                           f'{running} running · {time.time() - started:,.0f} s')

        with ThreadPoolExecutor(max_workers=workers) as pool:
            queue, pending, stop = iter(names), {}, False
            while True:
                while not stop and len(pending) < workers:
                    name = next(queue, None)
                    if name is None:
                        break
                    pending[pool.submit(run_one, name)] = name
                if not pending:
                    break
                done, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    name = pending.pop(fut)
                    try:
                        results[name] = fut.result()
                    except Exception as e:           # never lose the other replies to one crash
                        results[name] = dict(text='', stop_reason=None, served_by=None, usage={},
                                             seconds=0.0, error=f'{type(e).__name__}: {e}')
                    if results[name]['stop_reason'] is None:
                        stop = True                  # e.g. bad key / no credits: start no new requests
                show_progress(len(pending))

        failed = {n: r['error'] for n, r in results.items() if r['stop_reason'] is None}
        ok = {n: r for n, r in results.items() if r['stop_reason'] is not None}
        not_sent = [n for n in names if n not in results]
        if not ok:
            st.error('Request failed: ' + '; '.join(f'{n or "whole problem"}: {e}' for n, e in list(failed.items())[:3])
                     + ' — nothing was saved.')
            return
        if failed or not_sent:
            st.error(f'{len(failed)} request(s) failed and {len(not_sent)} were not sent (first error: '
                     f'{next(iter(failed.values()))}). The {len(ok)} replies that arrived are saved; '
                     'the rest count as unanswered.')
        for n, r in ok.items():                 # e.g. cut off at the token limit: saved, but flagged
            if r['error']:
                st.warning(f'{n or "Reply"}: {r["error"]}')
        served = sorted({r['served_by'] for r in ok.values() if r['served_by']})
        info = dict(info, model=model, mode=mode, prompt_version=PR.PROMPT_VERSION, served_by=', '.join(served),
                    requests=len(names), parallel=workers,
                    stop_reasons={n: r['stop_reason'] for n, r in ok.items()},
                    failed_parts=dict(failed, **{n: 'not sent' for n in not_sent}) or None,
                    input_tokens=sum(r['usage'].get('input_tokens') or 0 for r in ok.values()),
                    output_tokens=sum(r['usage'].get('output_tokens') or 0 for r in ok.values()),
                    reasoning_tokens=sum(r['usage'].get('reasoning_tokens') or 0 for r in ok.values()) or None,
                    seconds=sum(r['seconds'] for r in ok.values()),
                    wall_seconds=round(time.time() - started, 1))
        prompts = {n: PR.prompt_as_text(parts[n]['system'], parts[n]['user']) for n in ok}
        run_dir = R.save_run(problem_dir, label or default_label, info, prompts, {n: r['text'] for n, r in ok.items()})
        st.session_state['selected_run'] = run_dir.name
        m = R.load_run(run_dir)['metrics']
        st.success(f'Saved and scored as **{run_dir.name}** · variants answered: **{R.coverage_text(m)}** · '
                   f'recovery {fmt_pct(m["recovery_rate"])}. See tab 3 · Results.')

    # --- 2a) Claude through the Anthropic API --------------------------------------------
    if how == CLAUDE:
        c = st.columns([2, 1, 1])
        model = c[0].selectbox('Model', L.MODELS + ['other…'])
        if model == 'other…':
            model = c[0].text_input('Model ID', placeholder='claude-…')
        effort = c[1].selectbox('Effort', L.EFFORTS, index=L.EFFORTS.index('high'),
                                help='How hard the model thinks. Higher = slower, more tokens, usually better.')
        allow_fallback = c[2].checkbox('Allow fallback model', value=False,
                                       help='If the model declines, let the API answer with another model. '
                                            'Off by default: the answer would then not be this model\'s.')
        has_env_key = bool(os.environ.get('ANTHROPIC_API_KEY'))
        api_key = st.text_input('Anthropic API key', type='password',
                                placeholder='using ANTHROPIC_API_KEY from the environment' if has_env_key
                                else 'sk-ant-…  (or set ANTHROPIC_API_KEY before starting the app)',
                                help='Used for this request only; never saved.')
        st.caption(f'Sends {len(parts)} request(s) ({mode.lower()}). No tools are given to the model.')
        if st.button('🚀 Ask Claude', type='primary', disabled=blocked or not model):
            send_and_save(lambda s, u, cb: L.ask_claude(s, u, model, effort, api_key or None, allow_fallback, cb),
                          model, dict(source='claude-api', effort=effort, allow_fallback=allow_fallback),
                          f'{model}-{effort}')

    # --- 2b) Any model on OpenRouter (DeepSeek V4 Flash etc.) ------------------------------
    elif how == OPENROUTER:
        c = st.columns([2, 1, 1, 1])
        model = c[0].selectbox('Model', LO.MODELS + ['other…'],
                               help='deepseek/deepseek-v4-flash = DeepSeek V4 Flash. A dated ID (e.g. -0731) '
                                    'pins one exact version, so results stay comparable later.')
        if model == 'other…':
            model = c[0].text_input('OpenRouter model ID', placeholder='provider/model-name')
        effort = c[1].selectbox('Reasoning effort', LO.EFFORTS, index=LO.EFFORTS.index('high'),
                                help='How much the model thinks before answering. Reasoning is always on; '
                                     '"default" = the model\'s own default effort.')
        temperature = c[2].number_input('Temperature', 0.0, 2.0, 0.0, 0.1,
                                        help='0 = most repeatable. Higher = more varied answers.')
        seed = c[3].number_input('Seed', 0, 2**31 - 1, 0, help='Same seed + temperature 0 = as repeatable as '
                                                               'the provider allows.')
        api_key = st.text_input('OpenRouter API key', type='password',
                                placeholder='using OPENROUTER_API_KEY from the environment' if LO.api_key_from_env()
                                else 'sk-or-…  (or set OPENROUTER_API_KEY before starting the app)',
                                help='Used for this request only; never saved.')
        st.caption(f'Sends {len(parts)} request(s) ({mode.lower()}) to OpenRouter. No tools are given to the '
                   'model, and only providers that honour every setting above are used. Reasoning is always '
                   'on and there is no token limit; a reply without reasoning is rejected and retried on another host.')
        if st.button('🚀 Ask OpenRouter', type='primary', disabled=blocked or not model):
            send_and_save(lambda s, u, cb: LO.ask_openrouter(s, u, model, effort, api_key or None,
                                                             float(temperature), int(seed), cb),
                          model, dict(source='openrouter', effort=effort, temperature=float(temperature),
                                      seed=int(seed), max_tokens=None),
                          f'{model.split("/")[-1]}-{effort}')

    # --- 2b) Any LLM, by hand ------------------------------------------------------------
    else:
        st.markdown('1. Download the prompt in tab **1 · Prompt** and give it to the LLM.\n'
                    '2. Paste its reply below, or upload it as a .json / .txt file.')
        model_name = st.text_input('Which LLM answered?', placeholder='e.g. gpt-x, gemini-y, llama-z')
        up = st.file_uploader('Upload the reply', type=['json', 'txt'])
        pasted = st.text_area('…or paste the reply', height=200)
        reply = up.getvalue().decode('utf-8', errors='replace') if up else pasted
        if st.button('📥 Save and score', type='primary', disabled=not reply.strip()):
            info = dict(source='manual', model=model_name or 'unknown', effort=None, mode=ONE_REQUEST,
                        prompt_version=PR.PROMPT_VERSION, served_by=model_name or 'unknown')
            prompts = {'': PR.prompt_as_text(single['system'], single['user'])}
            run_dir = R.save_run(problem_dir, label or (model_name or 'manual'), info, prompts, {'': reply})
            st.session_state['selected_run'] = run_dir.name
            st.success(f'Saved and scored as **{run_dir.name}**. See tab 3 · Results.')


# ==============================================================================================
# Tab 3 — results of one run
# ==============================================================================================

with tab_results:
    saved = R.list_runs(problem_dir)
    if not saved:
        st.info('No runs yet for this problem. Get an answer in tab 2.')
    elif solution is None:
        st.info('Solve the problem in the generator app to score its runs.')
    else:
        run_names = [d.name for d in saved]
        default = st.session_state.get('selected_run')
        chosen = st.selectbox('Run', run_names, index=run_names.index(default) if default in run_names else 0)
        run_dir = saved[run_names.index(chosen)]

        # Scores are tied to the exact answer key they were computed with. If the problem was solved
        # again since, offer to re-score (from the saved reply, so the result is reproducible).
        run = R.load_run(run_dir)
        if run['metrics'].get('solution_fingerprint') != D.solution_fingerprint(problem_dir):
            st.warning('The answer key changed since this run was scored.')
        if st.button('🔁 Re-score from the saved reply',
                     help='Recomputes everything from the saved reply text. Same reply + same key = same numbers.'):
            R.rescore(problem_dir, run_dir)
            run = R.load_run(run_dir)
            st.success('Re-scored.')
        m, i = run['metrics'], run['info']

        st.caption(f"Model: **{i.get('served_by') or i.get('model')}**"
                   + (f" · effort **{i['effort']}**" if i.get('effort') else '')
                   + (f" · temperature {i['temperature']} · seed {i['seed']}" if 'temperature' in i else '')
                   + f" · source: {i.get('source')} · {i.get('mode', '')}"
                   + (f" ({i['requests']:,} requests, {i.get('parallel', 1)} at a time)" if i.get('requests', 1) > 1 else '')
                   + f" · prompt v{i.get('prompt_version', 1)}"
                   + (f" · {i['input_tokens']:,} in / {i['output_tokens']:,} out tokens" if i.get('input_tokens') else '')
                   + (f" ({i['reasoning_tokens']:,} reasoning)" if i.get('reasoning_tokens') else ''))
        for note in m.get('parse_notes', []):
            st.warning(f'Reading the reply: {note}')
        if i.get('failed_parts'):
            st.error(f"{len(i['failed_parts'])} of {i.get('requests', '?')} requests failed or were not sent; "
                     'their variants count as unanswered.')

        # Coverage first: how much of the problem the LLM answered at all (right or wrong).
        st.subheader('Coverage')
        if m.get('variants_total') is None:
            st.info('This run was scored before coverage was tracked. Press **Re-score** to see it.')
        else:
            c = st.columns(4)
            c[0].metric('Variants fully answered', f"{m['variants_complete']:,} / {m['variants_total']:,}",
                        help='Every removed genotype of the variant was answered (right or wrong).')
            c[1].metric('Partly answered', f"{m['variants_partial']:,}")
            c[2].metric('Not answered', f"{m['variants_missing']:,}")
            c[3].metric('Removed genotypes answered', f"{m['cells_answered']:,} / {m['cells']:,}")
            cov = run['coverage']
            if not cov.empty:
                # One square per variant, in data order: the whole picture at a glance.
                mark = {'complete': '🟩', 'partial': '🟨', 'missing': '⬜'}
                lines = [f"`{fam}` " + ''.join(mark[s_] for s_ in grp['status'])
                         for fam, grp in cov.groupby('family', sort=False)]
                st.markdown('  \n'.join(lines))
                st.caption('One square per variant, in data order: 🟩 fully answered · 🟨 partly · ⬜ not answered.')
                with st.expander('Coverage per variant'):
                    pick = st.multiselect('Show', ['complete', 'partial', 'missing'],
                                          default=['complete', 'partial', 'missing'], key='cov_pick')
                    st.dataframe(cov[cov['status'].isin(pick)], hide_index=True, use_container_width=True,
                                 height=280)

        st.subheader('Scores')

        # The three headline numbers.
        c = st.columns(3)
        c[0].metric('Validity rate', fmt_pct(m['validity_rate']), help=METRIC_HELP['validity'])
        c[1].metric('Uniqueness rate', fmt_pct(m['uniqueness_rate']), help=METRIC_HELP['uniqueness'])
        c[2].metric('Recovery rate', fmt_pct(m['recovery_rate']), help=METRIC_HELP['recovery'])

        # The counts behind them, so every number can be checked by hand.
        st.markdown(
            f"- **{m['suggested']:,}** hypotheses suggested · **{m['valid']:,}** valid · **{m['invalid']:,}** invalid "
            f"(impossible genotype {m['invalid_impossible']:,}, not asked {m['invalid_not_a_target']:,}, "
            f"malformed {m['invalid_malformed']:,})\n"
            f"- **{m['distinct']:,}** distinct · **{m['repeats']:,}** repeats\n"
            f"- **{m['distinct_valid']:,}** of **{m['admissible_space']:,}** admissible hypotheses found · "
            f"**{m['missed']:,}** missed\n"
            f"- Answered **{m['cells_answered']:,}** of **{m['cells']:,}** removed genotypes (person × variant)"
            + (f" · {m['conflict_cells']} have no consistent genotype" if m['conflict_cells'] else ''))

        with st.expander('How the rates are calculated'):
            st.markdown(
                'A **hypothesis** is one genotype proposed for one person at one variant. An answer '
                '`["RR", "Rr"]` for person 2 at variant 1:446 is two hypotheses.\n\n'
                '- **Validity** = valid suggestions ÷ all suggestions (repeats included).\n'
                '- **Uniqueness** = distinct suggestions ÷ all suggestions.\n'
                '- **Recovery** = distinct valid suggestions ÷ every admissible hypothesis in the answer key.\n\n'
                'Invalid reasons: **impossible** (that genotype cannot occur there), **not asked** (that '
                'person/variant was not removed, or does not exist), **malformed** (not two letters of the '
                "variant's gene). `rR`, `r/R` and `Rr` all count as the same genotype.")

        st.subheader('Per family')
        pf = run['per_family'].copy()
        if 'variants_total' in pf:
            pf.insert(1, 'coverage', 100.0 * pf['variants_complete'] / pf['variants_total'].where(pf['variants_total'] > 0))
        st.dataframe(pf, hide_index=True, use_container_width=True,
                     column_config={'coverage': st.column_config.ProgressColumn(
                         'variants fully answered', format='%.0f%%', min_value=0, max_value=100)})

        st.subheader('Every hypothesis')
        hyps = run['hypotheses']
        show = st.multiselect('Show verdicts', ['valid', 'impossible', 'not_a_target', 'malformed'],
                              default=['valid', 'impossible', 'not_a_target', 'malformed'])
        only_repeats = st.checkbox('Only repeats')
        view = hyps[hyps['verdict'].isin(show)]
        if only_repeats:
            view = view[view['repeat']]
        st.dataframe(view, hide_index=True, use_container_width=True, height=320)

        st.subheader('Missed admissible hypotheses')
        st.dataframe(run['missed'], hide_index=True, use_container_width=True, height=240)

        with st.expander('Raw reply'):
            replies = R.read_replies(run_dir)
            part = st.selectbox('Request', list(replies), format_func=lambda p: p or 'reply',
                                key='raw_part') if len(replies) > 1 else next(iter(replies), '')
            st.code((replies.get(part) or '(empty)')[:20000], language='json')
        st.caption(f'Saved in `{run_dir}`')


# ==============================================================================================
# Tab 4 — compare runs
# ==============================================================================================

with tab_compare:
    board = R.leaderboard(problem_dir)
    if board.empty:
        st.info('No runs yet for this problem.')
    else:
        st.dataframe(board, hide_index=True, use_container_width=True)
        st.caption('variants_covered = variants with every removed genotype answered / all variants to answer. '
                   'prompt_version: runs with different versions got different instructions. '
                   'up_to_date = scored with the current answer key. Re-score older runs in tab 3.')
        st.download_button('⬇️ Download as CSV', board.to_csv(index=False),
                           file_name=f'{problem_dir.name}_llm_runs.csv', mime='text/csv')
