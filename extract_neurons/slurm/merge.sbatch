#!/bin/bash
# Step 3: merge the per-rank checkpoints of collect into the map (one .npy per layer and list). CPU only.
# Submit from extract_neurons/, after collect has finished:
#   MODEL_NAME=gemma-3-4b-it sbatch --dependency=afterok:<COLLECT_JOBID> slurm/merge.sbatch
#SBATCH --job-name=nm-merge
#SBATCH --account=IscrC_VEC-ATCK
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# ---- edit these (MODEL_NAME and MAP_NAME can also be given on the command line, as above) ----
python="$HOME/miniconda3/envs/neuronexpl/bin/python"
MAPPINGS_DIR=/leonardo_work/IscrC_VEC-ATCK/mcontalb/neuronexpl/mappings
PARTIALS_ROOT=$SCRATCH/neuronmap
MODEL_NAME=${MODEL_NAME:-gemma-3-4b-it}
MAP_NAME=${MAP_NAME:-default}
# ---------------------

CORPUS_DIR=$MAPPINGS_DIR/$MODEL_NAME/$MAP_NAME/corpus
PARTIALS_DIR=$PARTIALS_ROOT/$MODEL_NAME/$MAP_NAME/partials
MAP_DIR=$MAPPINGS_DIR/$MODEL_NAME/$MAP_NAME/map

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "model $MODEL_NAME, map $MAP_NAME -> $MAP_DIR"
"$python" merge_and_export.py --partials_dir "$PARTIALS_DIR" --corpus_dir "$CORPUS_DIR" --out_dir "$MAP_DIR" --workers 16
