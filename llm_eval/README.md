# LLM Genotype Evaluation

A separate app that tests how well an LLM can work out hidden genotypes in a pedigree, and scores
it against the exact answer key.

## The idea

1. In the **Pedigree Dataset Generator** (`./run_ui.sh`), make a problem (some people's genotypes
   are removed) and **Solve** it. This writes `problem_dataset.json` (the puzzle) and
   `solution/solution_dataset.json` (every genotype each removed person could have).
2. In this app, give the puzzle to an LLM together with general instructions (the rules of
   Mendelian inheritance and the answer format). The LLM lists, for every removed genotype, every
   genotype that person could have.
3. The app scores each proposed genotype against the answer key.

## Running it

```bash
conda env create -f llm_eval/environment.yml     # once; Python 3.11 + streamlit + pandas + anthropic + openai
export OPENROUTER_API_KEY=sk-or-...              # to ask OpenRouter models (e.g. DeepSeek V4 Flash)
export ANTHROPIC_API_KEY=sk-ant-...              # to ask Claude
./run_llm_eval.sh                                # opens http://localhost:8502
```

(Keys can also be typed into the app instead; they are never saved.) If the environment already
exists from before OpenRouter support was added, update it with
`conda env update -f llm_eval/environment.yml`.

### DeepSeek V4 Flash (or any model) through OpenRouter

In tab **2 · Answer** choose **Ask a model on OpenRouter**, pick `deepseek/deepseek-v4-flash`
(or a dated version such as `deepseek/deepseek-v4-flash-0731`, which never changes, or type any
OpenRouter model ID), and press **Ask OpenRouter**.

- **Reasoning is always on.** **Reasoning effort** (default / low / medium / high) sets how much the
  model thinks; "default" uses the model's own default effort. A reply that comes back without any
  reasoning is rejected and retried with that host excluded (up to 3 tries). Reasoning tokens are
  saved with the run and shown in tab 3.
- **Temperature 0 + a fixed seed** (the defaults) make repeated runs as similar as the provider
  allows; both are saved with the run.
- **No token limit** is sent: the model may think and answer up to the host's own maximum output
  length.
- Only OpenRouter providers that support every setting (JSON schema, seed, temperature, reasoning)
  are used, so no setting is silently ignored.

### Big problems: how to send them (tab 1)

A model asked for a very long answer often stops early on purpose: it answers one or a few
variants, closes the JSON and ends normally (no "cut off" warning). Two things help:

- **The answer format is compact.** The model writes one entry per variant and groups the people
  who have the same list of possible genotypes:
  `{"family": "family_001", "variant": "1:446", "groups": [{"genotypes": ["RR", "Rr"], "individuals": ["2", "7"]}, …]}`.
  That is about 10× shorter than one entry per person (the format of prompt version 1, which is
  still read when scoring old runs or pasted answers).
- **Optionally, split the problem.** *One request per family*, or *One request per variant* (each
  request then contains one family's pedigree and a single variant, so every reply is short).
  **Requests sent at the same time** (tab 2, default 4) runs several requests in parallel. If a
  request fails outright (e.g. rate limit, no credits), no new ones are started, and the replies
  that did arrive are saved; the rest count as unanswered.

Tab 1 shows how many variants and removed genotypes the answer must cover, and the size of the
largest request.

### Coverage: how much of the problem was answered

While the requests run, tab 2 shows live **Variants answered so far: N / total**, read from the
streamed reply. Tab 3 then starts with **Coverage**: variants fully / partly / not answered,
removed genotypes answered, and one square per variant per family (🟩 complete · 🟨 partly ·
⬜ not answered). Tab 4 lists `variants_covered` for every run, so runs that stopped early stand
out at once. Coverage counts what was answered at all; whether it was right is what the three
rates measure.

### Time and cost

Every run records its **run time** (wall-clock, first request sent to last reply received), the
**request time** (each request's own duration summed; larger than the run time when requests ran
in parallel) and its **cost** in US dollars, plus time, cost and tokens per request. They are shown
at the top of tab 3 (with cost per variant answered) and as `run_seconds`, `request_seconds` and
`cost_usd` in tab 4, and tab 2 shows the time and cost so far while the requests run.

- **OpenRouter:** the cost OpenRouter charged, as it reports it with each reply (looked up from its
  generation record if a reply leaves it out). Attempts rejected for having no reasoning were paid
  for too and are included.
- **Claude:** estimated from the token counts and Anthropic's list prices in `llm.py` (`PRICES`,
  dated in `PRICES_AS_OF`; update them if prices change). Thinking is billed as output tokens.
- If some replies came back without a cost, the total is shown with a `+` (it is a lower bound).
  Runs saved before cost tracking show `—`; pasted answers have no time or cost.

Every run also records `prompt_version` (the instructions/answer format it was given), so runs made
with different instructions are not compared by accident.

Answers from any other LLM can be pasted or uploaded instead (tab **2 · Answer**), using the prompt
downloaded from tab **1 · Prompt**.

## How answers are scored

A **hypothesis** is one genotype proposed for one person at one variant. If person 2 could be `RR`
or `Rr` and person 5 only `rr`, the admissible hypotheses are (2, RR), (2, Rr) and (5, rr). An LLM
saying "person 5 could be `Rr`" made an invalid hypothesis.

| Rate | Formula | Question it answers |
|---|---|---|
| **Validity** | valid suggestions ÷ all suggestions (repeats included) | Of everything it said, how much was right? |
| **Uniqueness** | distinct suggestions ÷ all suggestions | Does it keep repeating itself? |
| **Recovery** | distinct valid suggestions ÷ all admissible hypotheses | How much of the full answer did it find? |

Invalid hypotheses are split into **impossible** (that genotype cannot occur there), **not asked**
(that person/variant was not removed, or does not exist) and **malformed** (not two letters of the
variant's gene). `rR`, `r/R` and `Rr` count as the same genotype.

Scoring is deterministic: the exact reply is saved with every run, and **Re-score** recomputes the
same numbers from it. (The LLM's own output is not deterministic — current Claude models do not
accept a temperature setting — which is why the reply itself is saved.)

## How leakage is prevented

- **Whitelist:** the prompt is built from scratch with only people (id, sex, father, mother) and
  variants (id, gene letter, genotypes with removed ones as null). Settings, the random seed, the
  source dataset's name and path, founder genome names, summaries and relationships are never copied.
- **Leak check:** before anything is sent, the app confirms against the answer key that every scored
  genotype is null in the prompt, that only whitelisted fields are present, and that the prompt does
  not mention the source dataset, seed or solution. Nothing can be sent if a check fails.
- **No tools:** the model (Claude or OpenRouter) gets no web, code execution or files, so the prompt is all it has.
- **No fallback by default:** a declined request is not silently answered by a different model.
- **Results stay out of the problem:** runs are saved in `llm_eval/runs/` (git-ignored), not in the
  problem folder, so a "Problem only" ZIP from the generator never contains graded answers.

## Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit page: sidebar + 4 tabs (Prompt, Answer, Results, Compare). |
| `data.py` | Finds problems in `outputs/`, loads the problem and answer key, fingerprints them. |
| `prompt.py` | Builds what the LLM sees, the instructions, the answer schema, and the leak check. |
| `llm.py` | Calls Claude (streaming, structured JSON output, no tools). |
| `llm_openrouter.py` | Calls any OpenRouter model, e.g. DeepSeek V4 Flash (same prompt, JSON schema, no tools, temperature + seed). |
| `scoring.py` | Reads a reply, turns it into hypotheses, computes the three rates. |
| `runs.py` | Saves runs, re-scores them, builds the comparison table. |

Each saved run (`llm_eval/runs/<problem>/<time>_<label>/`) holds the exact prompt and reply,
`metrics.json`, `hypotheses.csv` (every suggestion with its verdict), `per_family.csv`,
`coverage.csv` (per variant: removed genotypes answered, and complete / partial / missing) and
`missed.csv` (admissible hypotheses the LLM did not find). A split run has one
`prompt_<part>.txt` / `reply_<part>.txt` per request.
