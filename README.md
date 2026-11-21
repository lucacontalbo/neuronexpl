# Neuron maps (`extract_neurons/`) — how to run

Models: dense gated-MLP LLMs (Gemma 3, Qwen3.5, Mistral 3 / Ministral 3). A map is identified by the model and a
map name (`default` = general corpus; other names, e.g. `persuasion`, `bias`, use other corpora):
```
mappings/<MODEL_NAME>/<MAP_NAME>/corpus/   prepare_corpus.py output (configs/<MAP_NAME>.yaml, tokenized for the model)
mappings/<MODEL_NAME>/<MAP_NAME>/map/      merge_and_export.py output
$SCRATCH/neuronmap/<MODEL_NAME>/<MAP_NAME>/partials/   collect.py checkpoints
```
The sbatch files build these paths from `MODEL_NAME` (folder name of the model in `MODELS_DIR`) and `MAP_NAME`.

## 0. One-time setup
- Conda env with the packages in `requirements.txt`.
- Accept the terms of `bigcode/the-stack-smol` on Hugging Face (gated) and have an `HF_TOKEN`.
- Check `MODELS_DIR`, `MAPPINGS_DIR` and `--account` in `extract_neurons/slurm/*.sbatch`.
- `cd extract_neurons && mkdir -p logs`: submit everything from there.

## 1. Corpus (needs internet)
```bash
MODEL_NAME=gemma-3-4b-it MAP_NAME=default sbatch slurm/prepare_corpus.sbatch
```
Sources/budgets: `configs/default.yaml` (~100M tokens). Documents are split into 1024-token windows
(BOS + 1023 tokens if the tokenizer has a BOS, else 1024 tokens). The first token of each window and chat-template
tokens are never ranked (`mask.bin`). Output: `tokens.bin`, `mask.bin`, `windows.parquet`, `meta.json`.

## 2. Collect activations (offline)
```bash
MODEL_NAME=gemma-3-4b-it sbatch slurm/collect.sbatch                       # small: 4 tasks x 1 GPU
MODEL_NAME=gemma-3-27b-it GPUS_PER_RANK=2 sbatch --ntasks-per-node=2 --mem=200G slurm/collect.sbatch   # 24-27B
```
Checkpoints every 60 min. If the job stops, resubmit the same command: it resumes.
Throughput in `logs/nm-collect-*.out` (`tok/s`).

## 3. Merge + export
```bash
MODEL_NAME=gemma-3-4b-it sbatch --dependency=afterok:<COLLECT_JOBID> slurm/merge.sbatch
```

## Output
```
map/layer_XX/top_pos_pos.npy    gate>0, up>0      highest first
map/layer_XX/top_neg_neg.npy    gate<0, up<0      highest first
map/layer_XX/bottom_mixed.npy   opposite signs    most negative first
```
Each file is an array `[d_ff, K=1000]` of records `(doc_id, pos, activation)`: row = neuron, column = rank; empty slots
have `doc_id = -1`. Activation = `act(gate_proj(x)) * up_proj(x)` (the input of `down_proj`).
`pos` = token index inside the tokenized document `doc_id`, so same `doc_id` = same input.

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
