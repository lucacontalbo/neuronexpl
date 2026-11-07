#!/bin/bash
# Merge per-rank buffers and write the per-neuron files (CPU only).
#SBATCH --job-name=nm-merge
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# ---- edit these ----
CONDA_ENV=<your_env>
CORPUS_DIR=$WORK/neuronmap/corpus
PARTIALS_DIR=$SCRATCH/neuronmap/partials
MAP_DIR=$WORK/neuronmap/map
FORMAT=csv                             # or npy (~2.5x smaller)
# ---------------------

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
cd "$SLURM_SUBMIT_DIR"
python merge_and_export.py --partials_dir "$PARTIALS_DIR" --corpus_dir "$CORPUS_DIR" --out_dir "$MAP_DIR" \
    --format "$FORMAT" --workers 16
