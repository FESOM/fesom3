#!/bin/bash
# Run parallel scaling tests with JAX multi-CPU configuration
# Usage: bash scripts/run_parallel_test.sh [N_DEVICES]

set -e

# Default to 32 devices (all CPU cores)
N_DEVICES=${1:-32}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  JAX Parallel Scaling Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  CPU cores: $(nproc)"
echo "  JAX virtual devices: $N_DEVICES"
echo ""

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "jax081" ]; then
    echo -e "${BLUE}Activating conda environment: jax081${NC}"
    eval "$(conda shell.bash hook)"
    conda activate jax081
fi

# Configure JAX for CPU parallelism
export XLA_FLAGS="--xla_force_host_platform_device_count=$N_DEVICES"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo -e "${GREEN}Starting parallel scaling test...${NC}"
echo ""

# Run the example
python examples/06_parallel_scaling.py

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Test completed!${NC}"
echo -e "${GREEN}========================================${NC}"
