# Neuron map of gated-MLP LLMs — how to run

Supported: `google/gemma-3-4b-it` (GeGLU) and `Qwen/Qwen3.5-0.8B` (SwiGLU). The model is chosen by `MODEL_PATH` only.
Each model needs its **own** `CORPUS_DIR`, `PARTIALS_DIR`, `MAP_DIR` (token ids depend on the tokenizer).

## 0. One-time setup
- Conda env with the packages in `requirements.txt` (`pip install -r requirements.txt` inside the env).
- Accept the terms of `bigcode/the-stack-smol` on Hugging Face (gated) and have an `HF_TOKEN`.
- In `slurm/*.sbatch` fill the `<...>` placeholders: `--account`, `CONDA_ENV`, `MODEL_PATH`, and paths if needed.
- `mkdir -p logs` and submit everything from the repo root.

## 1. Corpus (needs internet)
```bash
sbatch slurm/prepare_corpus.sbatch      # partition lrd_all_serial (has internet)
```
Sources/budgets: `configs/corpus.yaml` (~100M tokens). Documents are split into 1024-token windows:
BOS + 1023 tokens for Gemma, 1024 tokens with no prefix for Qwen (which has no BOS).
The first token of each window and chat-template tokens are never ranked (`mask.bin`).
Output in `CORPUS_DIR`: `tokens.bin`, `mask.bin`, `windows.parquet`, `meta.json`.

## 2. Collect activations (offline, 1 node × 4 A100)
```bash
sbatch slurm/collect.sbatch
```
Checkpoints every 60 min to `PARTIALS_DIR`. If the job stops, resubmit the same script (same number of tasks): it resumes.
Check throughput in `logs/nm-collect-*.out` (`tok/s`).

## 3. Merge + export
```bash
sbatch --dependency=afterok:<COLLECT_JOBID> slurm/merge.sbatch
```
~1.04M files for Gemma (348,160 neurons × 3; Qwen3.5-0.8B: 86,016 neurons, ~258k files): check the inode quota first (`cindata`). `FORMAT=npy` is ~2.5× smaller than csv.

## Output
```
MAP_DIR/layer_XX/neuron_NNNNN/top_pos_pos.csv    gate>0, up>0      highest first
MAP_DIR/layer_XX/neuron_NNNNN/top_neg_neg.csv    gate<0, up<0      highest first
MAP_DIR/layer_XX/neuron_NNNNN/bottom_mixed.csv   opposite signs    most negative first
```
Rows: `rank,doc_id,pos,activation`, top/bottom 1000. Activation = `act(gate_proj(x)) * up_proj(x)` (input of `down_proj`; act = GELU for Gemma, SiLU for Qwen).
`pos` = token index inside the tokenized document `doc_id`, so same `doc_id` = same input.
Window = `pos // body`, with `body` = 1023 (Gemma) or 1024 (Qwen).

Inspect a neuron or a single link:
```bash
python show.py --corpus_dir $CORPUS_DIR --tokenizer $MODEL_PATH --map_dir $MAP_DIR --layer 12 --neuron 345 --list top_neg_neg -n 10
python show.py --corpus_dir $CORPUS_DIR --tokenizer $MODEL_PATH --doc 81234 --pos 77
```

# Neuron-guided perturbation (`run.py`)
Pushes a classifier (an LLM mapped above) toward `--target_label` by having a perturber LLM (OpenAI API, key in `.env`)
edit the input, using only the patterns of the classifier's neurons. Ground truth is never used.
```bash
python run.py --task asap2 --model Qwen/Qwen3.5-0.8B --map_dir $MAP_DIR --corpus_dir $CORPUS_DIR \
    --target_label 6 --k 3 --n_iter 5 --output results/qwen_t6.csv
```
- Neuron search (`neuron_search.py`): `S = g·(a* − a)`, pooled over the text tokens (`--pooling max|mean|topk`), ranked
  across layers, then re-ranked by the real ΔJ when patched to a* (`--no_causal_check` to skip). `--use_negative` also lowers neurons.
- Ablations: `--strategy negative` (most negative S) and `--strategy random`.
- The first run computes a* for every neuron and caches it in `$MAP_DIR/a_star_top10.pt`.
- Output columns: `id, perturbed_text, n_iterations, neurons, original_score, final_score, neuron_trace, score_trace`
  (the traces are JSON lists, one element per iteration).
- New task: add `tasks/<name>.py` and register it in `tasks/__init__.py`.
