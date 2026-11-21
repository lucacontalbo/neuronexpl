#!/bin/bash
# Step 2: record the strongest firings of every neuron of $MODEL_NAME over the corpus of $MAP_NAME. Offline.
# One srun task per GPUS_PER_RANK GPUs. Submit from extract_neurons/:
#   small models (<= 4B), 1 GPU per task:
#     MODEL_NAME=gemma-3-4b-it sbatch slurm/collect.sbatch
#   medium models (24-27B), the model is split over 2 GPUs per task:
#     MODEL_NAME=gemma-3-27b-it GPUS_PER_RANK=2 sbatch --ntasks-per-node=2 --mem=200G slurm/collect.sbatch
# Resumable: if the job stops, resubmit the same command (same number of tasks and GPUS_PER_RANK).
#SBATCH --job-name=nm-collect
#SBATCH --account=IscrC_VEC-ATCK
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# ---- edit these (MODEL_NAME, MAP_NAME and GPUS_PER_RANK can also be given on the command line, as above) ----
python="$HOME/miniconda3/envs/neuronexpl/bin/python"
MODELS_DIR=/leonardo_work/IscrC_VEC-ATCK/mcontalb/hf_models
MAPPINGS_DIR=/leonardo_work/IscrC_VEC-ATCK/mcontalb/neuronexpl/mappings
PARTIALS_ROOT=$SCRATCH/neuronmap
MODEL_NAME=${MODEL_NAME:-gemma-3-4b-it}
MAP_NAME=${MAP_NAME:-default}
GPUS_PER_RANK=${GPUS_PER_RANK:-1}
# ---------------------

CORPUS_DIR=$MAPPINGS_DIR/$MODEL_NAME/$MAP_NAME/corpus
PARTIALS_DIR=$PARTIALS_ROOT/$MODEL_NAME/$MAP_NAME/partials

set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
echo "model $MODEL_NAME, map $MAP_NAME, $SLURM_NTASKS tasks x $GPUS_PER_RANK GPUs -> $PARTIALS_DIR"
srun "$python" collect.py --model_path "$MODELS_DIR/$MODEL_NAME" --corpus_dir "$CORPUS_DIR" --out_dir "$PARTIALS_DIR" \
    --k 1000 --tokens_per_batch 16384 --ckpt_minutes 60 --gpus_per_rank "$GPUS_PER_RANK"
