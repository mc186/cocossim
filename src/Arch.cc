/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "Arch.h"
#include "memory.h"
#include "perf_enums.h"
#include "State.h"
#include <set>
#include <unordered_map>

std::unordered_map<int, int> state_updates;

State *Arch::have_idle_type(int ty) {
  for (auto &state: states) {
    if (state->get_ty_idx() == ty && state->get_state() == 0) {
      return state;
    }
  }
  return nullptr;
}

void Arch::init_waveforms() {
#ifdef VCD
#define dec(nm) fprintf(vcd, "$var wire %d %s %s_%d_%s $end\n", \
  WIDTH_ ##nm, \
  rand_chars[((vcd_idx << total_stat_bits) | (STAT_ ##nm ))], \
  state->get_ty_string().c_str(), vcd_idx, #nm);

  fprintf(vcd, "$timescale 1ns $end\n");
  fprintf(vcd, "$scope module top $end\n");
  fprintf(vcd, "$var wire 8 ? phase $end\n");
  for (int vcd_idx = 0; vcd_idx < states.size(); vcd_idx++) {
    auto &state = states.at(vcd_idx);
    dec(STATE);
    dec(IDLE_FROM_MEMORY);
    dec(JOB_IDX);
  }

  fprintf(vcd, "$upscope $end\n");
  fprintf(vcd, "$enddefinitions $end\n");
  fprintf(vcd, "$dumpvars\n");
  fprintf(vcd, "b00000000 ?\n");
  for (int i = 0; i < states.size(); ++i) {
    fprintf(vcd, "b%s %s\n", int_to_binary(0, 3).c_str(), rand_chars[(i << total_stat_bits) | STAT_STATE]);
  }
  fprintf(vcd, "$end\n");
#endif
}


RuntimeStats_t *Arch::get_cycles(TimeBasedEnqueue &time_enqueues) {
  int n_types;
  {
    std::set<int> a;

    for (const auto &st : states) {
        int ty_idx = st->get_ty_idx();
        a.insert(ty_idx);
    }
    n_types = (int) a.size();
  }
  
  n_idle_units = new int[n_types];
  memset(n_idle_units, 0, sizeof(int) * n_types);
  std::map<int, std::vector<Job *> *> task_frontiers;
  for (int i = 0; i < alloc_task_idx; ++i) {
    task_frontiers[i] = new std::vector<Job *>[n_types];
  }

  std::vector<Job *> *frontier[n_types];
  std::vector<int> task_ids(n_types);
  for (int i = 0; i < n_types; ++i) task_ids[i] = 0;
  std::function<void(Job *)> enqueue_job = [&](Job *job) -> void {
    task_frontiers[job->task_idx][job->get_type()].push_back(job);
    total_frontier += 1;
  };

  {
    int first_task = task_frontiers.begin()->first;
    for (int i = 0; i < n_types; ++i) {
      frontier[i] = &(task_frontiers[first_task][i]);
    }
  }

  if (time_enqueues.time_points.empty()) return nullptr;
  if (time_enqueues.time_points[0] != 0) {
    throw std::runtime_error("First time point must be 0");
  }
  auto *stats = new RuntimeStats_t[time_enqueues.to_enqueue.size()];
  auto total_states = states.size();
  for (int i = 0; i < time_enqueues.to_enqueue.size(); ++i) {
    stats[i].pct_active = new double[total_states];
  }

  uint64_t phase_cycles = 0;
  gcycles = 0;
  const uint64_t MAX_TIME = 0xFFFFFFFFFFFFFFFF;

  int phase_idx = 0;
  uint64_t next_phase;
  if (time_enqueues.time_points.size() > 1) {
    next_phase = time_enqueues.time_points[1];
  } else {
    next_phase = MAX_TIME;
  }

  auto &first_enqueue = time_enqueues.to_enqueue[0];


  for (auto &i: *first_enqueue) {
    enqueue_job(i);
  }

  int dram_cmds = 0;

  for (auto state: states) {
    n_idle_units[state->get_ty_idx()] += 1;
    total_idle += 1;
  }

  int logged_job_count = -1;


  auto *per_array_act = new uint64_t[states.size()];
  memset(per_array_act, 0, sizeof(uint64_t) * (states.size()));


  std::function<void(int)> write_stats = [&](int phase_idx) -> void {
    stats[phase_idx].cycles = phase_cycles;
    for (int i = 0; i < (states.size()); ++i) {
      stats[phase_idx].pct_active[i] = (double) (per_array_act[i] * 100) / (double) phase_cycles;
    }
  };

  double diff_accumulator_mem = 0;
  const double mem_slow_factor = 1;
  const double differential_mem = mem::dramsim3config->tCK / freq_sa / mem_slow_factor;
  const double cycle_adjust = 1. / freq_sa;

  while (!(total_idle == states.size() && total_frontier == 0)) {
    if (gcycles >= next_phase) {
      phase_idx++;
      state_updates.at(-1) = -1;
      for (auto *job: *(time_enqueues.to_enqueue[phase_idx])) {
        enqueue_job(job);
      }
      if (phase_idx + 1 < time_enqueues.time_points.size()) {
        next_phase = time_enqueues.time_points[phase_idx + 1];
      } else {
        next_phase = MAX_TIME;
      }
      write_stats(phase_idx - 1);

      phase_cycles = 0;
      memset(per_array_act, 0, sizeof(uint64_t) * (states.size()));
    }

    bool enqueued_job = false;
    
    for (int i = 0; i < n_types && !enqueued_job; ++i) {
      auto ty = i;
      auto &fr = frontier[i];
      if (fr->empty()) {
        task_ids[i] = (task_ids[i] + 1) % n_threads;
        fr = &(task_frontiers[task_ids[i]][i]);

        if (fr->empty()) {
          continue;
        }
      }
      
      Job *job = fr->front();
      if (n_idle_units[ty] > 0) {
        fr->erase(fr->begin());
        n_idle_units[ty] -= 1;
        total_idle -= 1;
        State *state = have_idle_type(ty);
        total_frontier--;
        state->j = job;
        LOG_TO_WAVEFORM(STAT_ID(JOB_IDX, state->vcd_idx), job->job_idx);
        state->init();
        enqueued_job = true;
      }
    }
#ifdef VCD
    bool first_state_update = true;
    for (auto pr: state_updates) {
      const int vcd_id = pr.first;
      const int to_state = pr.second;
      if (first_state_update) {
        first_state_update = false;
        fprintf(vcd, "#%f\n", (float) gcycles * cycle_adjust);
      }
      if (vcd_id >= 0) {
        int vcd_stat = STAT_EXTRACT(vcd_id);
        int width = STAT_WIDTH(vcd_stat);
        fprintf(vcd, "b%s %s\n", int_to_binary(to_state, width).c_str(), rand_chars[vcd_id]);
      } else {
        if (vcd_id == PHASE_STATE_IDX) {
          fprintf(vcd, "b%s ?\n", int_to_binary(phase_idx, 8).c_str());
        } else {
          throw std::exception();
        }
      }
      fflush(vcd);
    }
    state_updates.clear();
#endif

    gcycles++;
    phase_cycles++;

#if !defined(SILENCE) && !defined(DSE) || defined(DEBUG)
    if (logged_job_count != jobs_finished || gcycles % 100000 == 0) {
      logged_job_count = jobs_finished;
      printf("\rPHASE: %d, Cycles: %llu, Jobs finished: %d/%d, DRAM CMDs: %d", phase_idx, gcycles, jobs_finished, total_jobs, dram_cmds);
      mem::mem_sys->PrintStats();
      fflush(stdout);
    }
#endif
    diff_accumulator_mem += differential_mem;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "LoopDoesntUseConditionVariableInspection"
    while (diff_accumulator_mem >= differential_mem) {
      mem::mem_sys->ClockTick();
      diff_accumulator_mem -= 1;
    }
#pragma clang diagnostic pop

    for (int i = 0; i < states.size(); ++i) {
      bool is_active = states[i]->increment(enqueue_job, total_idle, n_idle_units);
      if (is_active) {
        per_array_act[i]++;
      }
    }

    bool successful_enqueue = true;
    for (int j = 0; j < dram_enq_per_cycle && successful_enqueue; ++j) {
      successful_enqueue = mem::try_enqueue_tx();
      dram_cmds += successful_enqueue;
    }
  }

#ifdef VCD
  bool first_state_update = true;
  for (auto pr: state_updates) {
    const int vcd_id = pr.first;
    const int to_state = pr.second;
    if (first_state_update) {
      first_state_update = false;
      fprintf(vcd, "#%f\n", (float) gcycles * cycle_adjust);
    }
    if (vcd_id >= 0) {
      int stat_extract = STAT_EXTRACT(vcd_id);
      fprintf(vcd, "b%s %s\n", int_to_binary(to_state, STAT_WIDTH(stat_extract)).c_str(), rand_chars[vcd_id]);
    } else {
      if (vcd_id == PHASE_STATE_IDX) {
        fprintf(vcd, "b%s ?\n", int_to_binary(phase_idx, 8).c_str());
      } else {
        throw std::exception();
      }
    }
  }
  state_updates.clear();
#endif

  printf("\rPHASE: %d, Cycles: %llu, Time: %fµs Jobs finished: %d/%d, DRAM CMDs: %d", phase_idx, gcycles, double(gcycles) * cycle_adjust / 1000, jobs_finished, total_jobs, dram_cmds);
  mem::mem_sys->PrintStats();
  fflush(stdout);
  write_stats(phase_idx);

  std::cout << std::endl;
  delete[] n_idle_units;
  return stats;
}
