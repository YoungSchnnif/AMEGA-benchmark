#!/bin/bash
#SBATCH --job-name=amega-benchmark
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/amega/log_%j.log
#SBATCH --error=logs/amega/log_%j.err
 
# ============================================================================
# AMEGA Benchmark für LLaMmlein auf HELMA
# 1 nodes x 4 GPUs = 8 GPUs
# ============================================================================

# ==== ENVIRONMENT SETUP ====
set -e

# Module laden
module load python/3.12-conda
module load cuda/12.8.0

# Conda aktivieren
source activate amega-benchmark

# ==== PATH & VARIABLES ====
PROJECT_DIR="/home/hpc/b185cb/b185cb28/AMEGA-benchmark"
DATA_DIR="$PROJECT_DIR/data"
MODEL_PATH="/hnvme/workspace/b185cb13-lsx/b185cb13-llammlein-1755918661/german_llms/gerInstruct/models/LLaMmlein7B_dpo_equal_de_eng_dpo_olmo/checkpoint-2927-copy2"
OUTPUT_DIR="$PROJECT_DIR/output"

# ==== ENVIRONMENT VARIABLES ====
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==== CREATE DIRECTORIES ====
mkdir -p $OUTPUT_DIR
mkdir -p logs

# ==== JOB INFO ====
echo "========================================"
echo "AMEGA Benchmark: LLaMmlein-7B"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# ==== NVIDIA GPU INFO ====
nvidia-smi

# ==== CHANGE TO PROJECT DIR ====
cd $PROJECT_DIR

# ==== RUN AMEGA BENCHMARK ====
echo "Starting AMEGA benchmark..."

python main.py \
    --data_dir $DATA_DIR \
    --generator_model_name_or_path $MODEL_PATH \
    --generator_type local \
    --case_id 1 \
    --output_dir $OUTPUT_DIR \
    --benchmark_mode generate \




echo "========================================"
echo "AMEGA benchmark complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Job finished at $(date)"
echo "========================================"