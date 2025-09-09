/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "EnqueueStructures.h"


void TimeBasedEnqueue::enqueue_at(uint64_t time, std::vector<Job*> *jobs) {
  time_points.push_back(time);
  to_enqueue.push_back(jobs);
}