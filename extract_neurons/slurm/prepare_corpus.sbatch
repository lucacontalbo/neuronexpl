#!/bin/bash
# Corpus download + tokenization. Needs internet: lrd_all_serial has it on Leonardo
# (alternatively run the srun line directly on a login node).
#SBATCH --job-name=nm-corpus
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# ---- edit these ----
CONDA_ENV=<your_env>
MODEL_PATH=<path/to/model>        # gemma-3-4b-it or Qwen3.5-0.8B
CORPUS_DIR=$WORK/neuronmap/corpus
export HF_HOME=$WORK/hf_cache          # streaming cache
# export HF_TOKEN=...                  # needed for bigcode/the-stack-smol (gated)
# ---------------------

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
cd "$SLURM_SUBMIT_DIR"
python prepare_corpus.py --config configs/corpus.yaml --tokenizer "$MODEL_PATH" --out_dir "$CORPUS_DIR"
