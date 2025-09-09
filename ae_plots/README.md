# COCOSSim Validation Plots

This directory contains validation results and comparison plots for COCOSSim.

## Files

- **`cocossim_vs_tpu_comparison.png`** - Validation comparison between COCOSSim simulation results and Google TPU v3 hardware measurements
- **`cocossim_vs_tpu_comparison.pdf`** - High-resolution PDF version of the validation plot

## Validation Methodology

COCOSSim was validated against Google TPU v3 hardware across diverse neural network workloads including:
- Matrix multiplication operations (GEMM)
- Convolutional neural networks (CNNs)  
- Transformer architectures
- Various activation functions and layer types

The validation demonstrates an average error rate of **13%** across all tested workloads, confirming COCOSSim's accuracy for architectural design space exploration.