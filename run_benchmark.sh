#!/bin/bash

# OLMo 2 Benchmarking Script for Dual B200 GPUs
# Usage: ./run_benchmark.sh <model_name> [revision] [additional_args...]

set -e

MODEL_NAME=$1
REVISION=$2
shift 2  # Remove first two arguments
ADDITIONAL_ARGS="$@"

if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name> [revision] [additional_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 allenai/OLMo-2-0425-1B"
    echo "  $0 allenai/OLMo-2-0425-1B stage1-step140000-tokens294B"
    echo "  $0 allenai/OLMo-2-1124-7B '' --batch_size 4 --max_questions 1000"
    echo ""
    echo "Available OLMo 2 models:"
    echo "  - allenai/OLMo-2-0425-1B"
    echo "  - allenai/OLMo-2-1124-7B" 
    echo "  - allenai/OLMo-2-1124-13B"
    echo "  - allenai/OLMo-2-0325-32B"
    echo ""
    echo "Model variants:"
    echo "  - Base: <model-name>"
    echo "  - SFT: <model-name>-SFT"
    echo "  - DPO: <model-name>-DPO"
    echo "  - Instruct: <model-name>-Instruct"
    exit 1
fi

echo "=== OLMo 2 Benchmark ==="
echo "Model: $MODEL_NAME"
[ -n "$REVISION" ] && echo "Revision: $REVISION"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)"
echo "========================="

# Check if we have multiple GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Running with $GPU_COUNT GPUs using torchrun..."
    
    # Optimal batch size for dual B200 setup
    BATCH_SIZE=${BATCH_SIZE:-16}  # Default to 16 for dual GPU
    
    TORCHRUN_ARGS="--nproc_per_node=$GPU_COUNT --master_port=29500"
    SCRIPT_ARGS="--model_name $MODEL_NAME --batch_size $BATCH_SIZE"
    
    if [ -n "$REVISION" ] && [ "$REVISION" != "''" ]; then
        SCRIPT_ARGS="$SCRIPT_ARGS --revision $REVISION"
    fi
    
    if [ -n "$ADDITIONAL_ARGS" ]; then
        SCRIPT_ARGS="$SCRIPT_ARGS $ADDITIONAL_ARGS"
    fi
    
    # Use torchrun for distributed inference
    torchrun $TORCHRUN_ARGS olmo2_benchmark.py $SCRIPT_ARGS
else
    echo "Running with single GPU..."
    
    BATCH_SIZE=${BATCH_SIZE:-8}  # Default to 8 for single GPU
    
    SCRIPT_ARGS="--model_name $MODEL_NAME --batch_size $BATCH_SIZE"
    
    if [ -n "$REVISION" ] && [ "$REVISION" != "''" ]; then
        SCRIPT_ARGS="$SCRIPT_ARGS --revision $REVISION"
    fi
    
    if [ -n "$ADDITIONAL_ARGS" ]; then
        SCRIPT_ARGS="$SCRIPT_ARGS $ADDITIONAL_ARGS"
    fi
    
    python olmo2_benchmark.py $SCRIPT_ARGS
fi

echo ""
echo "Benchmark completed! Check the benchmark_results/ directory for outputs."