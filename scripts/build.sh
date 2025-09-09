#!/bin/bash

# COCOSSim Build Script
# Copyright (c) 2025 APEX Lab, Duke University

set -e

echo "Building COCOSSim..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring build..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build the project
echo "Compiling..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build complete! Executable: build/perf_model"