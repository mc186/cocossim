# COCOSSim: TPU v3 Configuration

## Overview

**COCOSSim** is a cycle-accurate performance simulator for neural network accelerators. This branch contains the TPU v3 validated configuration with automatic splitting optimizations.

## Installation

### Prerequisites
- **C++17** compatible compiler (GCC 7+ or Clang 5+)
- **CMake 3.10** or higher
- **Git** for submodule management

### Quick Start
```bash
# Clone the repository
git clone https://github.com/mc186/cocossim.git
cd cocossim

# Initialize DRAMSim3 submodule
git clone --recursive https://github.com/umd-memsys/DRAMsim3.git dramsim3
# Alternative: use the setup script
./setup_dramsim.sh

# Build the project
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Simulation
```bash
# TPU v3 configuration simulation
./perf_model -c 1 -sa_sz 64 -sz_vu 64 -f 1 -i input.txt -o results.txt
```
 
## Features

This TPU v3 configuration includes:
- **Automatic workload parallelism** optimizations to match TPU performance 
- **Buffer size optimizations** for memory efficiency
- **Validated parameters** against Google TPU v3 hardware

## Validation Results

This configuration achieves **13% average error rate** when validated against Google TPU v3 hardware across diverse neural network workloads.