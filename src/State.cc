/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "Arch.h"
#include "State.h"
#include "global.h"

void State::enqueue_writes() {
  if (mem_write_left_unqueued > 0) {
    int to_enq = std::min(dram_enq_per_cycle, mem_write_left_unqueued);
    mem_write_left_unqueued -= to_enq;
    mem_queued += to_enq;
    for (int i = 0; i < to_enq; ++i) {
      to_enqueue.emplace_back(j->addr, true, core_memory_priority, this);
      j->addr += bytes_per_tx;
    }
  }
}

void State::enqueue_reads() {
  if (mem_read_left_unqueued > 0) {
    int to_enq = std::min(dram_enq_per_cycle, mem_read_left_unqueued);
    mem_read_left_unqueued -= to_enq;
    mem_queued += to_enq;
    for (int i = 0; i < to_enq; ++i) {
      to_enqueue.emplace_back(j->addr, false, core_memory_priority, this);
      j->addr += bytes_per_tx;
    }
  }
}

void State::check_idle_from_memory() {
  if (min_stage_cycles == 0 && !is_idle_from_memory &&
      (mem_read_left > 0 ||
       mem_write_left > 0)) {
    UPDATE_IDLEMEM(true);
  }
}

bool State::process_stage() {
  if (min_stage_cycles > 0)
    min_stage_cycles--;
  if (min_stage_cycles == 0 && mem_read_left == 0 && mem_write_left == 0) {
    return true;
  }
  check_idle_from_memory();
  return false;
}

void State::state_transfer(int st, int read_amt_bytes, int write_amt_bytes, int min_cycles) {
  IFVERB(printf("Time(%llu) - Transfer from %s to %s\n", gcycles, to_string(state), to_string(st)));
  UPDATE_STATE(st);
  min_stage_cycles = min_cycles;
  int rmin = read_amt_bytes > 0 ? 1 : 0;
  int wmin = write_amt_bytes > 0 ? 1 : 0;
  SET_READS(std::max(rmin, read_amt_bytes / bytes_per_tx));
  SET_WRITES(std::max(wmin, write_amt_bytes / bytes_per_tx));
  if (is_idle_from_memory) {
    UPDATE_IDLEMEM(false);
  }
}



static int g_vcd_ctr = 0;

State::State(int memory_priority) {
  core_memory_priority = memory_priority;
  vcd_idx = g_vcd_ctr++;
}

void vcd_stat_init(int vcd_idx, const char *sig_name) {
}
