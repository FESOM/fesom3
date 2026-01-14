#!/bin/bash
# Run test suite for jax_grids
# Usage: bash scripts/run_tests.sh [OPTIONS]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  JAX Grids Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate conda environment
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "jax081" ]; then
    echo -e "${BLUE}Activating conda environment: jax081${NC}"
    eval "$(conda shell.bash hook)"
    conda activate jax081
fi

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Install with: pip install pytest pytest-cov pytest-xdist"
    exit 1
fi

# Default options
MARKERS=""
PARALLEL=""
COVERAGE=""
VERBOSE="-v"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            MARKERS="-m cpu"
            shift
            ;;
        --gpu)
            MARKERS="-m gpu"
            shift
            ;;
        --no-slow)
            MARKERS="-m 'not slow'"
            shift
            ;;
        --parallel)
            PARALLEL="-n auto"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=core --cov=utilities --cov-report=html --cov-report=term"
            shift
            ;;
        --help)
            echo "Usage: bash scripts/run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu         Run only CPU tests"
            echo "  --gpu         Run only GPU tests"
            echo "  --no-slow     Skip slow tests"
            echo "  --parallel    Run tests in parallel"
            echo "  --coverage    Generate coverage report"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Run tests
echo -e "${GREEN}Running tests...${NC}"
echo "Options: $MARKERS $PARALLEL $COVERAGE $VERBOSE"
echo ""

pytest tests/ $MARKERS $PARALLEL $COVERAGE $VERBOSE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  All tests passed!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Some tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
