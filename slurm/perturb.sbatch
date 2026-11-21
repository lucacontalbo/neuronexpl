#!/bin/bash
# Neuron-guided perturbation of the samples of $TASK: defender $DEFENDER_NAME (transformers, gradients + hooks) on
# the GPUs $DEFENDER_GPUS, attacker gpt-oss-20b served by vLLM on GPU 0. One array task per neuron strategy:
#   0 best   = --strategy gradient   (highest S, causal check)
#   1 worst  = --strategy negative   (most negative S: ablation)
#   2 random = --strategy random     (ablation)
# Resumable: resubmitting skips the samples already in the output csv.
# Submit from the repo root: sbatch slurm/perturb.sbatch
# Medium defenders (14-31B): DEFENDER_GPUS="1,2" and --gres=gpu:3. All defenders run in bfloat16 (run.py default).
#SBATCH --account=IscrC_VEC-ATCK
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=perturb
#SBATCH --array=0-2
#SBATCH --output=./outputs/tmp_%A_%a.out

START_TIME=$(date +%s)

set -euo pipefail
set -x
PS4='+ $(date "+%F %T") [line ${LINENO}] '
trap 'rc=$?; echo "ERROR at line ${LINENO}: ${BASH_COMMAND} (exit code: ${rc})"' ERR

# ---- edit these ----
python="$HOME/miniconda3/envs/neuronexpl/bin/python"          # env with requirements.txt (transformers >= 5.2)
SINGULARITY_IMAGE="/leonardo_work/IscrC_CFXBench/mcontalb/mtqaautogen/containers/scratch_local/$USER/vllm_sandbox_171"
BIND_DIR="/leonardo_work/IscrC_VEC-ATCK/mcontalb"             # must contain the attacker model
MODELS_DIR="/leonardo_work/IscrC_VEC-ATCK/mcontalb/hf_models"
MAPPINGS_DIR="/leonardo_work/IscrC_VEC-ATCK/mcontalb/neuronexpl/mappings"   # extract_neurons/ output
TASK="asap2"                   # asap2, biggen or judgebench
DEFENDER_NAME="Qwen3.5-0.8B"   # folder name in MODELS_DIR and MAPPINGS_DIR
MAP_NAME="default"             # which neuron map of the defender: MAPPINGS_DIR/DEFENDER_NAME/MAP_NAME
ATTACKER_PATH="$MODELS_DIR/gpt-oss-20b"
ATTACKER_NAME="gpt-oss-20b"
TARGET_LABEL=6
K=3                    # new neurons per iteration
N_ITER=5               # max perturbation rounds per sample
LIMIT="500"            # test only the first N samples; empty = all
EXTRA_ARGS="--include_source_text"   # e.g. "--include_source_text" (asap2 only) or "--use_negative"
VLLM_GPU=0             # GPU of the attacker (vLLM), index within the job's allocation
DEFENDER_GPUS="1"      # GPU(s) of the defender (run.py): "1" for small models, "1,2" for 24-27B models
# ---------------------

DEFENDER_PATH="$MODELS_DIR/$DEFENDER_NAME"
MAP_DIR="$MAPPINGS_DIR/$DEFENDER_NAME/$MAP_NAME/map"
CORPUS_DIR="$MAPPINGS_DIR/$DEFENDER_NAME/$MAP_NAME/corpus"

strategies=(gradient negative random)
names=(best worst random)
strategy=${strategies[$SLURM_ARRAY_TASK_ID]}
name=${names[$SLURM_ARRAY_TASK_ID]}
out="${TASK}_${DEFENDER_NAME}_${MAP_NAME}_${ATTACKER_NAME}_t${TARGET_LABEL}_k${K}_n${N_ITER}_${name}"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

export CACHE_ROOT="$PROJECT_DIR/.cache"
mkdir -p "$CACHE_ROOT"/{vllm,huggingface,torch,xdg,triton}
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/huggingface"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_HOME="$CACHE_ROOT/torch"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export VLLM_ENGINE_READY_TIMEOUT_S="3600"
export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_TORCH_HOME="$TORCH_HOME"
export SINGULARITYENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export SINGULARITYENV_VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S"
export SINGULARITYENV_VLLM_ENABLE_CUDA_COMPATIBILITY=1
export SINGULARITYENV_TIKTOKEN_ENCODINGS_BASE=/leonardo_work/IscrC_VEC-ATCK/mcontalb/hf_models/tiktoken_encodings

HOST="127.0.0.1"
PORT=$("$python" -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")

echo "Strategy: $strategy ($name) -> results/${out}.csv"
for f in "$DEFENDER_PATH" "$ATTACKER_PATH" "$MAP_DIR" "$CORPUS_DIR"; do
    if [[ ! -e "$f" ]]; then
        echo "'$f' not found."
        exit 1
    fi
done
nvidia-smi -L

cleanup() {
    rc=$?
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server PID ${VLLM_PID}"
        kill "${VLLM_PID}" || true
        wait "${VLLM_PID}" || true
    fi
    echo "EXIT with code ${rc}"
}
trap cleanup EXIT

mkdir -p ./outputs/vllm ./outputs/perturb ./results

# vLLM has GPU $VLLM_GPU to itself; the defender gets all of GPU(s) $DEFENDER_GPUS (forward + backward passes).
echo "Starting vLLM..."
CUDA_VISIBLE_DEVICES="$VLLM_GPU" SINGULARITYENV_CUDA_VISIBLE_DEVICES="$VLLM_GPU" singularity exec --nv \
    --bind "$BIND_DIR:$BIND_DIR" \
    "$SINGULARITY_IMAGE" \
    /bin/bash -lc "
        if command -v python >/dev/null 2>&1; then
            VLLM_PY=python
        elif command -v python3 >/dev/null 2>&1; then
            VLLM_PY=python3
        elif [[ -x /opt/venv/bin/python ]]; then
            VLLM_PY=/opt/venv/bin/python
        else
            echo 'No python found inside container'
            exit 1
        fi

        exec \"\$VLLM_PY\" -m vllm.entrypoints.openai.api_server \
            --model \"$ATTACKER_PATH\" \
            --served-model-name \"$ATTACKER_NAME\" \
            --host \"$HOST\" \
            --port \"$PORT\" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.9 \
            --max-model-len 32768
    " > "./outputs/vllm/perturb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to become ready..."
for i in $(seq 1 720); do
    if curl -fsS "http://${HOST}:${PORT}/v1/models" > /dev/null; then
        echo "vLLM is ready."
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM process exited before becoming ready."
        break
    fi
    sleep 5
done
if ! curl -fsS "http://${HOST}:${PORT}/v1/models" | grep -q "\"${ATTACKER_NAME}\""; then
    echo "vLLM did not start correctly."
    tail -n 100 "./outputs/vllm/perturb_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" || true
    exit 1
fi

N_DEFENDER_GPUS=$(echo "$DEFENDER_GPUS" | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES="$DEFENDER_GPUS" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$python" run.py \
    --task "$TASK" \
    --gpus "$N_DEFENDER_GPUS" \
    --model "$DEFENDER_PATH" \
    --map_dir "$MAP_DIR" \
    --corpus_dir "$CORPUS_DIR" \
    --perturber_model "$ATTACKER_NAME" \
    --perturber_base_url "http://${HOST}:${PORT}/v1" \
    --target_label "$TARGET_LABEL" \
    --k "$K" \
    --n_iter "$N_ITER" \
    --strategy "$strategy" \
    --seed 0 \
    --output "results/${out}.csv" \
    ${LIMIT:+--limit "$LIMIT"} $EXTRA_ARGS

echo "Total time: $(( $(date +%s) - START_TIME )) seconds"

mv "$PROJECT_DIR/outputs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" \
   "$PROJECT_DIR/outputs/perturb/${out}_${SLURM_ARRAY_JOB_ID}.out"
