#!/bin/bash
# Step 1: download and tokenize the corpus configs/$MAP_NAME.yaml for the model $MODEL_NAME.
# Needs internet: lrd_all_serial has it on Leonardo. Submit from extract_neurons/:
#   MODEL_NAME=gemma-3-4b-it MAP_NAME=default sbatch slurm/prepare_corpus.sbatch
#SBATCH --job-name=nm-corpus
#SBATCH --account=IscrC_VEC-ATCK
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# ---- edit these (MODEL_NAME and MAP_NAME can also be given on the command line, as above) ----
python="$HOME/miniconda3/envs/neuronexpl/bin/python"
MODELS_DIR=/leonardo_work/IscrC_VEC-ATCK/mcontalb/hf_models                # one folder per model
MAPPINGS_DIR=/leonardo_work/IscrC_VEC-ATCK/mcontalb/neuronexpl/mappings
MODEL_NAME=${MODEL_NAME:-gemma-3-4b-it}  # folder name in MODELS_DIR
MAP_NAME=${MAP_NAME:-default}            # corpus config: configs/$MAP_NAME.yaml
export HF_HOME=$WORK/hf_cache            # streaming cache
# export HF_TOKEN=...                    # needed for bigcode/the-stack-smol (gated)
# ---------------------

CORPUS_DIR=$MAPPINGS_DIR/$MODEL_NAME/$MAP_NAME/corpus

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "model $MODEL_NAME, map $MAP_NAME -> $CORPUS_DIR"
"$python" prepare_corpus.py --config "configs/$MAP_NAME.yaml" --tokenizer "$MODELS_DIR/$MODEL_NAME" --out_dir "$CORPUS_DIR"
