#!/bin/bash
# Run example scripts for jax_grids
# Usage: bash scripts/run_example.sh <example_name> [OPTIONS]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if example name provided
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No example specified${NC}"
    echo ""
    echo "Usage: bash scripts/run_example.sh <example_file> [OPTIONS]"
    echo ""
    echo "Available examples:"
    echo "  01_simple_quad_halo.py       - Basic quad grid with halo exchange"
    echo "  02_triangle_halo.py          - Triangle grid halo exchange"
    echo "  03_mixed_grid_halo.py        - Mixed grid complexity"
    echo "  04_staggered_grid.py         - Different staggering types"
    echo "  05_multi_width_halo.py       - Variable halo widths"
    echo ""
    echo "Example:"
    echo "  bash scripts/run_example.sh examples/01_simple_quad_halo.py"
    exit 1
fi

EXAMPLE_FILE=$1
shift  # Remove first argument

# Check if file exists
if [ ! -f "$EXAMPLE_FILE" ]; then
    # Try prepending examples/ if not found
    if [ -f "examples/$EXAMPLE_FILE" ]; then
        EXAMPLE_FILE="examples/$EXAMPLE_FILE"
    else
        echo -e "${RED}Error: Example file not found: $EXAMPLE_FILE${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Running JAX Grids Example${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Example: ${EXAMPLE_FILE}${NC}"
echo ""

# Activate conda environment
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "jax081" ]; then
    echo -e "${BLUE}Activating conda environment: jax081${NC}"
    eval "$(conda shell.bash hook)"
    conda activate jax081
fi

# Set JAX configuration for CPU/GPU
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}

# Display system info
echo -e "${BLUE}System Information:${NC}"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  JAX platform: $JAX_PLATFORMS"
echo "  CPU cores: $(nproc)"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
echo ""

# Run the example
echo -e "${GREEN}Running example...${NC}"
echo ""

python "$EXAMPLE_FILE" "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Example completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Example failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
