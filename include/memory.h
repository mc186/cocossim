/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_MEMORY_H
#define PROSE_COMPILER_MEMORY_H
#include "State.h"
#include "memory_system.h"


namespace mem {
  bool try_enqueue_tx();
  using mem_ty = dramsim3::JedecDRAMSystem;

  extern dramsim3::Config *dramsim3config;
  extern std::unordered_map<uint64_t, State *> address_reads_bkwds_lookup, address_writes_bkwds_lookup;
  extern mem_ty *mem_sys;

  void setup();
};// namespace mem
#endif//PROSE_COMPILER_MEMORY_H
