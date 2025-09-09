/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_ARRAYSTATE_H
#define PROSE_COMPILER_ARRAYSTATE_H

#include "Job.h"
#include "perf_enums.h"
#include <functional>
#include <set>

#include "Arch.h"

#ifdef VCD
#define LOG_TO_WAVEFORM(stat_idx, to) \
  state_updates[stat_idx] = int(to);
#define UPDATE_STATE(x)                                  \
  state_updates[STAT_ID(STATE, vcd_idx)] = int(x); \
  set_state(x);
#define UPDATE_IDLEMEM(to)                                      \
  state_updates[STAT_ID(IDLE_FROM_MEMORY, vcd_idx)] = to; \
  is_idle_from_memory = to
#else
#define LOG_TO_WAVEFORM(stat_idx, to)
#define UPDATE_STATE(x) set_state(x)
#define UPDATE_IDLEMEM(to)
#endif

#ifdef VERBOSE
#define IFVERB(x) x
#else
#define IFVERB(x)
#endif

#define SET_READS(x) mem_read_left_unqueued = mem_read_left = x
#define SET_WRITES(x) mem_write_left = mem_write_left_unqueued = x

#define TO_IDLE_CLEANUP()                                   \
  jobs_finished++;                                          \
  total_idle++;                                             \
  n_idle_units[get_ty_idx()] += 1;                \
  for (auto *child: j->children) {                          \
    child->rem_deps -= 1;                                   \
    if (child->rem_deps == 0) {                             \
      IFVERB(std::cout << "enqueuing child " << std::endl); \
      enqueue_job(child);                                   \
    }                                                       \
  }                                                         \
  j->is_done = true;                                        \
  j = nullptr



using enqueue_job_f_t = std::function<void(Job *)>;

struct State {
  int sz;          // Size of the functional array
  Job *j = nullptr;// Job being processed by the array

  uint8_t vcd_idx = 0;// Index for VCD tracing

  int min_stage_cycles;       // Minimum cycles required to read data / shift / whatever
  int mem_read_left;          // Remaining memory reads to complete
  int mem_write_left;         // Remaining memory writes to complete
  int mem_read_left_unqueued; // Unqueued memory reads left
  int mem_write_left_unqueued;// Unqueued memory writes left
  int mem_queued;             // Memory operations queued
  int core_memory_priority;
  bool is_idle_from_memory = false;

  int loop_row_tiles; // Number of row tiles in the loop
  int loop_cols_tiles;// Number of column tiles in the loop
  int row_i, col_i;   // Current row and column indices

  int beats_per_wb;// Number of memory beats per write-back

  bool activation_in_buffer = false;// Enable DRAM-to-buffer flow simulation if false

  virtual ~State() = default;
  State() = delete;
  State(int memory_priority);


  void enqueue_writes();
  void enqueue_reads();
  void check_idle_from_memory();
  bool process_stage();
  void state_transfer(int st, int read_amt, int write_amt, int min_cycles);
  virtual void init() = 0;
  virtual bool increment(const enqueue_job_f_t &, int &total_idle, int *n_idle_units) = 0;
  virtual void set_state(int st) = 0;
  virtual int get_state() = 0;

  virtual int get_ty_idx() = 0;
  virtual std::string get_ty_string() = 0;
};

void vcd_stat_init(int vcd_idx, const char *sig_name);

#endif// PROSE_COMPILER_ARRAYSTATE_H
