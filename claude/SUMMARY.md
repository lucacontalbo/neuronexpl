# Work summary (last update: 2026-11-09)

## 1. Starting point
- `extract_neurons/` (unchanged): builds the offline **neuron map** of a gated-MLP LLM (Gemma-3-4B-it, Qwen3.5-0.8B).
  A neuron = one unit of the MLP of one decoder layer, its activation is the input of `mlp.down_proj`,
  `act(gate_proj(x)) * up_proj(x)`. For every neuron the map stores the top-1000 corpus firings in three lists:
  `top_pos_pos` (gate>0, up>0), `top_neg_neg` (gate<0, up<0), `bottom_mixed` (opposite signs, most negative).
- The previous online analysis (`main.py`, `models.py`, `perturb_verify.py`, `task_datasets.py`, `prompts/`) was deleted
  on request (2026-11-07).

## 2. New system: neuron-guided perturbation (`run.py`)
Goal: flip a **defending classifier** toward a fixed `--target_label` by having a **perturber LLM** edit the input,
using only the patterns of the classifier's neurons. Ground truth is never used.

Per sample:
1. The defender predicts a label; if it is already the target, the sample is skipped (not written).
2. Up to `--n_iter` times:
   - pick `--k` new neurons (never reused within a sample) with the neuron search;
   - fetch the top-10 corpus contexts of each neuron (120 tokens before, 40 after, firing token in `[[ ]]`);
   - the perturber edits the text: (i) to move the defender toward the target, (ii) using ONLY those patterns,
     (iii) keeping the text otherwise identical;
   - the edited text is classified again; stop as soon as the target is reached.

### Files
| file | role |
|---|---|
| `run.py` | CLI, per-sample loop, csv output (appended per sample; rerun = resume) |
| `classifier.py` | defender: label = argmax of the label-token logits (greedy); activations/gradients via `down_proj` pre-hooks; patched forward for the causal check |
| `neuron_search.py` | a* table, neuron scoring, causal check, contexts (reuses `extract_neurons/show.py:load_list`) |
| `perturber.py` | perturber prompt; OpenAI API or any OpenAI-compatible server (`--perturber_base_url`, e.g. vLLM) |
| `tasks/` | one file per task (`asap2.py`); register new tasks in `tasks/__init__.py` |
| `slurm/perturb.sbatch` | Leonardo job: vLLM attacker + defender on one GPU, 3 strategies as an array |

### Neuron search (agreed design)
- Objective: J = log P(target) − log P(current) (= logit difference). g = dJ/da at each essay token.
- Score: **S = g · (a\* − a)**, an estimate of ΔJ when the neuron moves to a naturally observed value a\*.
  - direction "+" (raise, default): a\* = mean of the top-10 activations in `top_pos_pos`, contexts from `top_pos_pos`;
  - direction "−" (lower, only with `--use_negative`): a\* = mean of the 10 most negative in `bottom_mixed`, contexts from `bottom_mixed`.
  - a\* is computed once per map and cached in `<map_dir>/a_star_top10.pt`.
- Only essay tokens are scored (the perturber can edit only those).
- Pooling over tokens: `--pooling max` (default) | `mean` | `topk` (`--pool_topk`).
- Layer localization: none (option A): global ranking across layers, S is already in J units.
- **Causal check** (default on, `--causal_candidates 50`): the top candidates are patched to a\* at their scoring
  positions and re-ranked by the real ΔJ; neurons with ΔJ > 0 are kept; if none helps, the highest ΔJ are taken anyway.
  `--no_causal_check` ranks by S only (ablation).
- Strategies: `gradient` (best), `negative` (ablation: most negative S, still asked to raise), `random` (ablation).
- Temperatures: defender greedy; perturber 0.5, except gpt-5/o-series which only accept 1.

### Output csv columns
`id, perturbed_text, n_iterations, neurons ("layer:neuron;..."), original_score, final_score, neuron_trace, score_trace`
- `neuron_trace`: JSON list, one dict per iteration; keys `"layer:neuron"` of all neurons used so far, values
  `{dir, since, prev, now, a_star, gap_closed}` with prev/now = activation pooled as in the search, before/after the edit,
  `gap_closed = (now − prev) / (a_star − prev)`.
- `score_trace`: JSON list, one dict per iteration: `{J_prev, J_now, score_now}` (J w.r.t. the label predicted before the edit).

## 3. Task: ASAP 2.0 automatic essay scoring (`tasks/asap2.py`)
- Test set: `data/asap2/ASAP_2_Final_github_test.csv` (from https://github.com/scrosseye/ASAP_2.0, zip password
  `asap2_test`), 7421 essays, 5 prompts, scores 1–6.
- Prompt: assignment + (optional, `--include_source_text`) source text + holistic rubric (1–6) + essay + "answer with the score only".
- Source texts: `data/asap2/source_texts/*.txt`, transcribed by hand from the scanned PDFs of the repo
  (no text layer), verified page by page; the student-instruction pages are excluded.

## 4. Experiments (Leonardo)
`sbatch slurm/perturb.sbatch` (fill the `<...>` paths first): defender Qwen3.5-0.8B, attacker gpt-oss-20b on vLLM
(60% of the GPU), array task 0/1/2 = best/worst/random neurons, outputs in
`results/asap2_qwen3.5-0.8b_gpt-oss-20b_t<target>_k<k>_n<n_iter>_{best,worst,random}.csv`.
To test only the first N essays: `LIMIT=N` in the sbatch (= `--limit N` of `run.py`).

## 5. Status / tests
- Smoke-tested on CPU with Qwen3.5-0.8B, fake a\*/contexts and a stub perturber: all strategies, pooling options,
  `--use_negative`, causal check on/off, both trace columns.
- Perturber tested against a stub OpenAI-compatible server (request + `<text>` parsing).
- Not yet run: real neuron map, real vLLM/gpt-oss-20b, GPU.
