/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_RUNTIMESTATS_T_H
#define PROSE_COMPILER_RUNTIMESTATS_T_H
#include <cstdint>

struct RuntimeStats_t {
  uint64_t cycles;
  double *pct_active;
};

#endif//PROSE_COMPILER_RUNTIMESTATS_T_H