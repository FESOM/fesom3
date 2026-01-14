#!/bin/bash
#SBATCH --job-name=jax_grids          # Job name
#SBATCH --partition=gpu               # Partition (queue) - adjust as needed
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs (adjust as needed)
#SBATCH --time=01:00:00               # Time limit (HH:MM:SS)
#SBATCH --mem=64G                     # Memory per node
#SBATCH --output=slurm-%j.out         # Standard output log
#SBATCH --error=slurm-%j.err          # Standard error log

# SLURM batch submission script for JAX grids on HPC
# This is a TEMPLATE - adjust parameters based on your HPC system
#
# Usage:
#   sbatch scripts/submit_slurm.sh <example_or_script.py>
#
# Example:
#   sbatch scripts/submit_slurm.sh examples/01_simple_quad_halo.py

set -e  # Exit on error

# ============================================
# Configuration
# ============================================

# Get the script to run from command line argument
if [ $# -lt 1 ]; then
    echo "Error: No script specified"
    echo "Usage: sbatch scripts/submit_slurm.sh <script.py>"
    exit 1
fi

SCRIPT_TO_RUN=$1

# Print job information
echo "=========================================="
echo "JAX Grids HPC Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Script: $SCRIPT_TO_RUN"
echo "=========================================="
echo ""

# ============================================
# Environment Setup
# ============================================

# Load required modules (adjust for your HPC system)
# module load anaconda3
# module load cuda/12.0
# module load gcc/11.2.0

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate jax081

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Set JAX configuration
export JAX_PLATFORMS=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# For multi-GPU (if needed in future)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# ============================================
# System Information
# ============================================

echo "System Information:"
echo "  Hostname: $(hostname)"
echo "  CPUs: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
    echo ""
fi

# ============================================
# Run the Script
# ============================================

echo "Starting computation..."
echo "Time: $(date)"
echo ""

# Run the Python script
python "$SCRIPT_TO_RUN"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
