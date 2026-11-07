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
conda env create -f llm_eval/environment.yml     # once; Python 3.11 + streamlit + pandas + anthropic
export ANTHROPIC_API_KEY=sk-ant-...              # only needed to ask Claude from the app
./run_llm_eval.sh                                # opens http://localhost:8502
```

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
- **No tools:** Claude gets no web, code execution or files, so the prompt is all it has.
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
| `scoring.py` | Reads a reply, turns it into hypotheses, computes the three rates. |
| `runs.py` | Saves runs, re-scores them, builds the comparison table. |

Each saved run (`llm_eval/runs/<problem>/<time>_<label>/`) holds the exact prompt and reply,
`metrics.json`, `hypotheses.csv` (every suggestion with its verdict), `per_family.csv` and
`missed.csv` (admissible hypotheses the LLM did not find).
