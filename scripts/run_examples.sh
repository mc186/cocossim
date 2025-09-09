#!/bin/bash

# COCOSSim Example Runner
# Copyright (c) 2025 APEX Lab, Duke University

set -e

# Check if build exists
if [ ! -f "build/perf_model" ]; then
    echo "Error: COCOSSim not built. Run 'scripts/build.sh' first."
    exit 1
fi

EXAMPLES_DIR="examples"
RESULTS_DIR="results"

# Create results directory
mkdir -p $RESULTS_DIR

echo "Running COCOSSim examples..."

# Run simple matrix multiplication
echo "1. Simple Matrix Multiplication"
./build/perf_model -c 1 -sa_sz 64 -vu_sz 64 -f 1 \
    -i $EXAMPLES_DIR/simple_matmul.txt \
    -o $RESULTS_DIR/simple_matmul_results.txt

# Run CNN model
echo "2. CNN Model"
./build/perf_model -c 1 -sa_sz 64 -vu_sz 64 -f 1 \
    -i $EXAMPLES_DIR/cnn_model.txt \
    -o $RESULTS_DIR/cnn_results.txt

# Run transformer model
echo "3. Transformer Model"
./build/perf_model -c 1 -sa_sz 64 -vu_sz 64 -f 1 \
    -i $EXAMPLES_DIR/basic_transformer.txt \
    -o $RESULTS_DIR/transformer_results.txt

# Compare different dataflow modes
echo "4. Dataflow Comparison (Output Stationary vs Weight Stationary)"
./build/perf_model -c 1 -sa_sz 64 -vu_sz 64 -ws 0 -f 1 \
    -i $EXAMPLES_DIR/simple_matmul.txt \
    -o $RESULTS_DIR/os_dataflow_results.txt

./build/perf_model -c 1 -sa_sz 64 -vu_sz 64 -ws 1 -f 1 \
    -i $EXAMPLES_DIR/simple_matmul.txt \
    -o $RESULTS_DIR/ws_dataflow_results.txt

echo "Examples completed! Results saved in $RESULTS_DIR/"