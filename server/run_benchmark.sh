#!/bin/bash

# Benchmark Runner Script for ArtMancer
# Usage: ./run_benchmark.sh <input_path> [options]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INPUT_PATH=""
TASK_TYPE="object-removal"
PROMPT="remove the object"
SAMPLE_COUNT=0
NUM_INFERENCE_STEPS=40
GUIDANCE_SCALE=1.0
TRUE_CFG_SCALE=4.0
NEGATIVE_PROMPT=""
SEED=""
INPUT_QUALITY="high"
OUTPUT_DIR=""

# Function to print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  ./run_benchmark.sh <input_path> [options]"
    echo ""
    echo -e "${BLUE}Arguments:${NC}"
    echo "  <input_path>          Path to benchmark folder or ZIP file"
    echo "                       Must contain: input/, mask/, groundtruth/ folders"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -t, --task TYPE       Task type: object-removal (default)"
    echo "  -p, --prompt TEXT     Prompt for all images (required)"
    echo "  -s, --samples N       Number of samples (0 = all, default: 0)"
    echo "  --steps N             Number of inference steps (default: 40)"
    echo "  --guidance-scale F    Guidance scale (default: 1.0)"
    echo "  --cfg-scale F         True CFG scale (default: 4.0)"
    echo "  --negative-prompt T   Negative prompt"
    echo "  --seed N              Random seed"
    echo "  --quality TYPE        Input quality: super_low, low, medium, high, original (default: high)"
    echo "  -o, --output DIR      Output directory for results (default: ./benchmark_results)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  ./run_benchmark.sh ./benchmark_data.zip -p 'remove the object'"
    echo "  ./run_benchmark.sh ./benchmark_data/ -p 'remove object' -s 10 --steps 50"
    echo "  ./run_benchmark.sh ./data.zip -p 'remove' --quality high -o ./results"
}

# Parse arguments
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

INPUT_PATH="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--task)
            TASK_TYPE="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -s|--samples)
            SAMPLE_COUNT="$2"
            shift 2
            ;;
        --steps)
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --guidance-scale)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --cfg-scale)
            TRUE_CFG_SCALE="$2"
            shift 2
            ;;
        --negative-prompt)
            NEGATIVE_PROMPT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --quality)
            INPUT_QUALITY="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Get script directory (must be done early for path resolution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate input path
if [ -z "$INPUT_PATH" ]; then
    echo -e "${RED}Error: Input path is required${NC}"
    print_usage
    exit 1
fi

# Convert to absolute path if relative
# Check if path is absolute (starts with /)
if [ "${INPUT_PATH#/}" != "$INPUT_PATH" ]; then
    # Already absolute path
    ABS_INPUT_PATH="$INPUT_PATH"
else
    # Relative path - try multiple locations
    # First, try from current working directory
    if [ -e "$INPUT_PATH" ]; then
        ABS_INPUT_PATH="$(cd "$(dirname "$INPUT_PATH")" 2>/dev/null && pwd)/$(basename "$INPUT_PATH")"
    # Try from script directory
    elif [ -e "$SCRIPT_DIR/$INPUT_PATH" ]; then
        ABS_INPUT_PATH="$(cd "$SCRIPT_DIR" && cd "$(dirname "$INPUT_PATH")" 2>/dev/null && pwd)/$(basename "$INPUT_PATH")"
    # Try from parent directory (in case running from server/)
    elif [ -e "../$INPUT_PATH" ]; then
        ABS_INPUT_PATH="$(cd .. && cd "$(dirname "$INPUT_PATH")" 2>/dev/null && pwd)/$(basename "$INPUT_PATH")"
    else
        # Last resort: construct from current directory
        ABS_INPUT_PATH="$(pwd)/$INPUT_PATH"
    fi
fi

# Check if path exists
if [ ! -e "$ABS_INPUT_PATH" ]; then
    echo -e "${RED}Error: Input path does not exist: $INPUT_PATH${NC}"
    echo ""
    echo -e "${YELLOW}Debug information:${NC}"
    echo "  Original path:  $INPUT_PATH"
    echo "  Absolute path:  $ABS_INPUT_PATH"
    echo "  Current dir:    $(pwd)"
    echo "  Script dir:     $SCRIPT_DIR"
    echo ""
    echo -e "${YELLOW}Searching for similar paths...${NC}"
    # Try to find similar paths
    if command -v find &> /dev/null; then
        FOUND=$(find . -type d -name "$(basename "$INPUT_PATH")" 2>/dev/null | head -3)
        if [ -n "$FOUND" ]; then
            echo "  Found similar directories:"
            echo "$FOUND" | sed 's/^/    /'
        fi
    fi
    echo ""
    echo -e "${YELLOW}Available in current directory:${NC}"
    ls -la . 2>/dev/null | head -10 | sed 's/^/  /' || echo "  (cannot list directory)"
    exit 1
fi

# Update INPUT_PATH to absolute path for display
INPUT_PATH="$ABS_INPUT_PATH"

# Validate prompt
if [ -z "$PROMPT" ] || [ "$PROMPT" = "remove the object" ] && [ "$TASK_TYPE" != "white-balance" ]; then
    echo -e "${YELLOW}Warning: Using default prompt 'remove the object'${NC}"
fi

# Set default output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./benchmark_results"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/app/cli/benchmark_cli.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python CLI script not found: $PYTHON_SCRIPT${NC}"
    echo "Please ensure the script exists or run from the server directory."
    exit 1
fi

# Check if uv is available
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ArtMancer Benchmark Runner${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Input Path:      $INPUT_PATH"
echo "  Task Type:       $TASK_TYPE"
echo "  Prompt:         $PROMPT"
echo "  Sample Count:   $SAMPLE_COUNT (0 = all)"
echo "  Inference Steps: $NUM_INFERENCE_STEPS"
echo "  Guidance Scale: $GUIDANCE_SCALE"
echo "  CFG Scale:      $TRUE_CFG_SCALE"
echo "  Input Quality:  $INPUT_QUALITY"
echo "  Output Dir:     $OUTPUT_DIR"
[ -n "$NEGATIVE_PROMPT" ] && echo "  Negative Prompt: $NEGATIVE_PROMPT"
[ -n "$SEED" ] && echo "  Seed:            $SEED"
echo ""

# Run benchmark
echo -e "${BLUE}Starting benchmark...${NC}"
echo ""

cd "$SCRIPT_DIR"

$PYTHON_CMD "$PYTHON_SCRIPT" \
    --input "$ABS_INPUT_PATH" \
    --task "$TASK_TYPE" \
    --prompt "$PROMPT" \
    --samples "$SAMPLE_COUNT" \
    --steps "$NUM_INFERENCE_STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --cfg-scale "$TRUE_CFG_SCALE" \
    --quality "$INPUT_QUALITY" \
    --output "$OUTPUT_DIR" \
    $([ -n "$NEGATIVE_PROMPT" ] && echo "--negative-prompt \"$NEGATIVE_PROMPT\"") \
    $([ -n "$SEED" ] && echo "--seed $SEED")

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Benchmark completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
else
    echo ""
    echo -e "${RED}❌ Benchmark failed with exit code: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi

