#!/bin/bash
#SBATCH --job-name=amega-helma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/amega_helma_%j.log
#SBATCH --error=logs/amega_helma_%j.err

# =============================================================
# AMEGA Benchmark - HELMA Cluster (CPU-only, local generator)
# Generator : local LLaMmlein7B checkpoint
# Evaluator : openai/gpt-oss-120b via hub.nhr.fau.de gateway
# =============================================================

echo "==================================================="
echo "AMEGA Job Started at $(date)"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "==================================================="

# -----------------------------------------------------------
# 1. Environment setup
# -----------------------------------------------------------
echo "=== Environment Setup ==="

module load python/3.12-conda
echo "✓ Conda module loaded"

source activate amega-benchmark
echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"

AMEGA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AMEGA_DIR"
echo "✓ Working directory: $(pwd)"

export PYTHONPATH="$AMEGA_DIR:${PYTHONPATH:-}"
echo "✓ PYTHONPATH set: $PYTHONPATH"

# Temporary directories
export TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"
echo "✓ Temporary directory: $TMPDIR"
echo "✓ HuggingFace cache  : $HF_HOME"

mkdir -p logs results
echo "✓ Output directories created (logs/, results/)"

# -----------------------------------------------------------
# 2. GPU / node information
# -----------------------------------------------------------
echo "=== Node Information ==="

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node IP address : $NODE_IP"

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "GPUs detected   : $GPU_COUNT"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
        | while IFS= read -r line; do echo "  GPU $line"; done
else
    echo "GPUs detected   : 0 (nvidia-smi not available — CPU-only run)"
fi

# -----------------------------------------------------------
# 3. Load API keys from .env
# -----------------------------------------------------------
echo "=== API Key Setup ==="

if [ -f "$AMEGA_DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$AMEGA_DIR/.env"
    set +a
    echo "✓ Environment variables loaded from .env"
else
    echo "✗ ERROR: .env file not found at $AMEGA_DIR/.env"
    exit 1
fi

# -----------------------------------------------------------
# 4. Pre-flight checks
# -----------------------------------------------------------
echo "=== Pre-flight Checks ==="

# Override MODEL_PATH by setting AMEGA_MODEL_PATH in the environment before submitting.
MODEL_PATH="${AMEGA_MODEL_PATH:-/hnvme/workspace/b185cb13-lsx/b185cb13-llammlein-1755918661/german_llms/gerInstruct/models/LLaMmlein7B_dpo_equal_de_eng_dpo_olmo/checkpoint-2927-copy}"

if [ -d "$MODEL_PATH" ]; then
    echo "✓ Generator model found : $MODEL_PATH"
else
    echo "✗ ERROR: Generator model not found at $MODEL_PATH"
    exit 1
fi

echo "Checking required data files..."
for file in data/cases.csv data/questions.csv data/sections.csv data/criteria.csv; do
    if [ -f "$file" ]; then
        echo "✓ $file ($(wc -l < "$file") lines)"
    else
        echo "✗ ERROR: Required data file not found: $file"
        exit 1
    fi
done

# -----------------------------------------------------------
# 5. System information
# -----------------------------------------------------------
echo "=== System Information ==="

echo "Hostname     : $(hostname)"
echo "Python       : $(python --version 2>&1)"
echo "Torch        : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "Transformers : $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"

# -----------------------------------------------------------
# 6. Run AMEGA Benchmark
# -----------------------------------------------------------
echo "==================================================="
echo "Running AMEGA Benchmark"
echo "  case_id        : 1   (change to a number, range e.g. 1-5, or 'all')"
echo "  benchmark_mode : full"
echo "  n_eval         : 3"
echo "  k_eval         : 1"
echo "==================================================="

python main.py \
    --data_dir ./data \
    --output_dir ./results \
    --case_id 1 \
    --generator_model_name_or_path "$MODEL_PATH" \
    --generator_type local \
    --evaluator_model_name_or_path "openai/gpt-oss-120b" \
    --evaluator_type openai \
    --n_eval 3 \
    --k_eval 1 \
    --benchmark_mode full

EXIT_CODE=$?

# -----------------------------------------------------------
# 7. Report job status
# -----------------------------------------------------------
echo "==================================================="
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "✓ AMEGA Job COMPLETED SUCCESSFULLY at $(date)"
    echo "Results saved to: $AMEGA_DIR/results/"
    ls -lah results/
else
    echo "✗ AMEGA Job FAILED at $(date) (exit code: $EXIT_CODE)"
    echo "Check the error log: logs/amega_helma_${SLURM_JOB_ID}.err"
fi
echo "==================================================="

exit "$EXIT_CODE"
