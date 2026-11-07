#!/bin/bash
# Activation collection: one task per A100. Fully offline. Resumable: resubmitting
# continues from the last checkpoint of each rank (keep the same number of tasks).
#SBATCH --job-name=nm-collect
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# ---- edit these ----
CONDA_ENV=<your_env>
MODEL_PATH=<path/to/model>        # gemma-3-4b-it or Qwen3.5-0.8B
CORPUS_DIR=$WORK/neuronmap/corpus
PARTIALS_DIR=$SCRATCH/neuronmap/partials
# ---------------------

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
srun python collect.py --model_path "$MODEL_PATH" --corpus_dir "$CORPUS_DIR" --out_dir "$PARTIALS_DIR" \
    --k 1000 --tokens_per_batch 16384 --ckpt_minutes 60
