#!/bin/bash

# Benchmark Runner Script for ArtMancer
# Usage: ./run_benchmark.sh <input_path> [options]

set -euo pipefail  # Exit on error, unset vars, fail in pipelines

WORK_DIR=""

cleanup() {
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    fi
    trap - EXIT
}

trap cleanup EXIT

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
DEVICE_MODE="auto"

# Function to print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  ./run_benchmark.sh <input_path> [options]"
    echo ""
    echo -e "${BLUE}Arguments:${NC}"
    echo "  <input_path>          Path to dataset folder or ZIP file."
    echo "                       Expecting sub-folders: input/, mae/ (optional), mask/, target/"
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
    echo "  --device MODE         Device preference: auto, cuda, xpu, mps, cpu (default: auto)"
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
        --device)
            DEVICE_MODE="$2"
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
DATASET_PREPROCESSOR="$SCRIPT_DIR/app/utils/dataset_preprocessor.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python CLI script not found: $PYTHON_SCRIPT${NC}"
    echo "Please ensure the script exists or run from the server directory."
    exit 1
fi

if [ ! -f "$DATASET_PREPROCESSOR" ]; then
    echo -e "${RED}Error: Dataset preprocessor not found: $DATASET_PREPROCESSOR${NC}"
    exit 1
fi

# Resolve Python interpreter
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Prefer running under uv if available
if command -v uv &> /dev/null; then
    PYTHON_CMD=(uv run "$PYTHON_BIN")
else
    PYTHON_CMD=("$PYTHON_BIN")
fi

# Prepare temp workspace
WORK_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'benchmark_tmp')"
MANIFEST_PATH="$WORK_DIR/dataset_manifest.json"
NORMALIZED_DATASET="$WORK_DIR/dataset"

echo -e "${BLUE}🔧 Preparing dataset structure (input/mae/mask/target) ...${NC}"
if ! PREPROCESS_JSON=$("${PYTHON_CMD[@]}" "$DATASET_PREPROCESSOR" \
        --source "$INPUT_PATH" \
        --output "$NORMALIZED_DATASET" \
        --manifest "$MANIFEST_PATH"); then
    echo -e "${RED}❌ Dataset preprocessing failed${NC}"
    exit 1
fi
INPUT_PATH="$NORMALIZED_DATASET"

# Extract summary fields from JSON output
TOTAL_PAIRS="$(
    PREPROCESS_JSON="$PREPROCESS_JSON" "$PYTHON_BIN" - <<'PY'
import json, os
data = json.loads(os.environ["PREPROCESS_JSON"])
print(data.get("total_pairs", 0))
PY
)"

MAE_AVAILABLE="$(
    PREPROCESS_JSON="$PREPROCESS_JSON" "$PYTHON_BIN" - <<'PY'
import json, os
data = json.loads(os.environ["PREPROCESS_JSON"])
print("yes" if data.get("mae_available") else "no")
PY
)"

echo -e "${GREEN}✅ Dataset normalized (${TOTAL_PAIRS} pairs, mae:$MAE_AVAILABLE). Manifest: $MANIFEST_PATH${NC}"

# Select device backend
DEVICE_SELECTION_OUTPUT=$(
    REQUESTED_DEVICE="$DEVICE_MODE" "${PYTHON_CMD[@]}" - <<'PY'
import os, sys, torch

requested = os.environ["REQUESTED_DEVICE"].strip().lower()
valid_modes = {"auto", "cuda", "xpu", "mps", "cpu"}

def available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if name == "mps":
        backend = getattr(torch.backends, "mps", None)
        return backend.is_available() if backend else False
    if name == "cpu":
        return True
    return False

if requested not in valid_modes:
    sys.stderr.write(
        f"Invalid --device value '{requested}'. Valid options: auto, cuda, xpu, mps, cpu.\\n"
    )
    sys.exit(1)

order = ("cuda", "xpu", "mps", "cpu")

if requested == "auto":
    for name in order:
        if available(name):
            print(name)
            sys.exit(0)
    print("cpu")
    sys.exit(0)

if not available(requested):
    sys.stderr.write(
        f"Requested device '{requested}' is not available on this system.\\n"
    )
    sys.exit(2)

print(requested)
PY
)
DEVICE_STATUS=$?
if [ $DEVICE_STATUS -ne 0 ]; then
    echo -e "${RED}❌ Unable to select execution device. Aborting.${NC}"
    exit 1
fi

SELECTED_DEVICE="$(echo "$DEVICE_SELECTION_OUTPUT" | tr -d '\r\n')"
export ARTMANCER_DEVICE="$SELECTED_DEVICE"

case "$SELECTED_DEVICE" in
    cuda)
        echo -e "${GREEN}🟢 CUDA device detected. Running with NVIDIA acceleration.${NC}"
        ;;
    xpu)
        export SYCL_DEVICE_FILTER="gpu"
        export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
        echo -e "${GREEN}🟢 Intel XPU detected. Running with torch.xpu backend.${NC}"
        ;;
    mps)
        echo -e "${GREEN}🟢 Apple MPS backend detected.${NC}"
        ;;
    cpu)
        export CUDA_VISIBLE_DEVICES=""
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        echo -e "${YELLOW}⚠️  No compatible GPU detected. Falling back to CPU.${NC}"
        ;;
    *)
        echo -e "${YELLOW}⚠️  Unknown device '$SELECTED_DEVICE'. Proceeding but behavior may be undefined.${NC}"
        ;;
esac

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
echo "  Device:         $SELECTED_DEVICE"
echo "  Output Dir:     $OUTPUT_DIR"
[ -n "$NEGATIVE_PROMPT" ] && echo "  Negative Prompt: $NEGATIVE_PROMPT"
[ -n "$SEED" ] && echo "  Seed:            $SEED"
echo ""

# Run benchmark
echo -e "${BLUE}Starting benchmark...${NC}"
echo ""

cd "$SCRIPT_DIR"

"${PYTHON_CMD[@]}" "$PYTHON_SCRIPT" \
    --input "$INPUT_PATH" \
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
    if [ -f "$MANIFEST_PATH" ]; then
        cp "$MANIFEST_PATH" "$OUTPUT_DIR/benchmark_manifest.json"
        echo -e "${GREEN}📄 Dataset manifest saved to $OUTPUT_DIR/benchmark_manifest.json${NC}"
    fi
    echo -e "${GREEN}✅ Benchmark completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
else
    echo ""
    echo -e "${RED}❌ Benchmark failed with exit code: $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi

