#!/bin/bash
# Part 1 grid: every defender x task x target label, one perturb.sbatch array job (best, worst, random) each.
# Run from the repo root:
#   bash slurm/submit_grid.sh                          # the whole grid (6 x 13 = 78 jobs, 234 runs)
#   bash slurm/submit_grid.sh gemma-4-31B-it           # only the jobs of one defender
#   bash slurm/submit_grid.sh gemma-4-31B-it biggen    # ... and one task
#   bash slurm/submit_grid.sh gemma-4-31B-it biggen 5  # ... and one target (e.g. to resubmit it)
# Runs are resumable: resubmitting skips the samples already in the output csv.
set -euo pipefail

ONLY_DEFENDER=${1:-}
ONLY_TASK=${2:-}
ONLY_TARGET=${3:-}

# defender -> its GPUs (the attacker, vLLM, always has GPU 0)
declare -A DEFENDER_GPUS=(
    [Qwen3.5-0.8B]="1"
    [gemma-4-E4B-it]="1"
    [Ministral-3-3B-Instruct-2512-BF16]="1"
    [Ministral-3-14B-Instruct-2512-BF16]="1"
    [Qwen3.5-27B]="1,2"
    [gemma-4-31B-it]="1,2,3"
)
DEFENDERS=(Qwen3.5-0.8B gemma-4-E4B-it Ministral-3-3B-Instruct-2512-BF16
           Ministral-3-14B-Instruct-2512-BF16 Qwen3.5-27B gemma-4-31B-it)

# task -> its target labels
declare -A TARGETS=(
    [asap2]="1 2 3 4 5 6"
    [biggen]="1 2 3 4 5"
    [judgebench]="A B"
)
TASKS=(asap2 biggen judgebench)

mkdir -p outputs results
for defender in "${DEFENDERS[@]}"; do
    [[ -n "$ONLY_DEFENDER" && "$defender" != "$ONLY_DEFENDER" ]] && continue
    gpus=${DEFENDER_GPUS[$defender]}
    n_gpus=$(( $(echo "$gpus" | tr ',' '\n' | wc -l) + 1 ))
    for task in "${TASKS[@]}"; do
        [[ -n "$ONLY_TASK" && "$task" != "$ONLY_TASK" ]] && continue
        for target in ${TARGETS[$task]}; do
            [[ -n "$ONLY_TARGET" && "$target" != "$ONLY_TARGET" ]] && continue
            jid=$(TASK=$task DEFENDER_NAME=$defender DEFENDER_GPUS=$gpus TARGET_LABEL=$target \
                  sbatch --parsable --gres=gpu:$n_gpus --job-name="p-${defender}-${task}-t${target}" slurm/perturb.sbatch)
            echo "$jid  $defender  $task  target $target  ($n_gpus GPUs)"
        done
    done
done
