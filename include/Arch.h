/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_ARCH_H
#define PROSE_COMPILER_ARCH_H

#include "EnqueueStructures.h"
#include "RuntimeStats_t.h"
#include "global.h"
#include <unordered_map>
#include <map>
#include <vector>

extern std::unordered_map<int, int> state_updates;

struct Arch {
  std::vector<State *> states;

  State *have_idle_type(int ty);


  int total_frontier = 0;
  int *n_idle_units = nullptr;
  int total_idle = 0;

  Arch() = default;

  void init_waveforms();

  RuntimeStats_t *get_cycles(TimeBasedEnqueue &time_enqueues);

  private:
  bool have_inited = false;
};

#endif//PROSE_COMPILER_ARCH_H
