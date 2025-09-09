/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_ENQUEUESTRUCTURES_H
#define PROSE_COMPILER_ENQUEUESTRUCTURES_H

#include <vector>
#include "Job.h"

struct TimeBasedEnqueue {
  std::vector<uint64_t> time_points;
  std::vector<std::vector<Job*>*> to_enqueue;
  void enqueue_at(uint64_t time, std::vector<Job*> *jobs);
};
#endif//PROSE_COMPILER_ENQUEUESTRUCTURES_H
