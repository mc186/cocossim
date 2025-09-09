/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "units/standard/VectorUnit.h"
#include "frontends/standard/StandardUnits.h"
#include "global.h"
#include "State.h"
#include "perf_enums.h"

using namespace VectorUnit;

bool VecUnitState::increment(const std::function<void(Job *)> &enqueue_job, int &total_idle, int *n_idle_units) {
  auto *sj = (VecUnitJob *) j;
  int lin, par;
  switch (state) {
    case VectorUnit::VPUState::unbuffered_lin:
    case VectorUnit::VPUState::unbuffered_par:
    case VectorUnit::VPUState::buffered_lin:
    case VectorUnit::VPUState::buffered_par:
      // Process vector operations with linearized and parallel dimensions
      lin = sj->linearized_dimension;
      par = sj->parallel_dimension;

      enqueue_reads();
      if (process_stage()) {
        auto &ph_ar = sj->phases;
        if (ph_ar.empty()) {
          // All phases completed, write results
          state_transfer(VectorUnit::VPUState::write,
                         0,
                         lin * par * data_type_width * batch_size,
                         0);
        } else if (ph_ar.front().first == VPUPhase::REDUCE) {
          // Reduction phase: compute along linear dimension
          state_transfer(VectorUnit::VPUState::buffered_lin,
                         0, 0,
                         ph_ar.front().second * lin * div_ru(par, sz));
          ph_ar.pop();
        } else if (ph_ar.front().first == VPUPhase::BROADCAST) {
          // Broadcast phase: distribute values across parallel dimension
          state_transfer(VectorUnit::VPUState::buffered_par,
                         0, 0,
                         div_ru(lin * par * ph_ar.front().second, sz));
          ph_ar.pop();
        }
      }
      break;
    case VectorUnit::VPUState::write:
      enqueue_writes();
      if (process_stage()) {
        state_transfer(VectorUnit::idle, 0, 0, 0);
        TO_IDLE_CLEANUP();
      }
      break;
    case VectorUnit::VPUState::idle:
      break;
    default:
      std::cerr << "Reached an unexpected state..." << std::endl;
      throw std::exception();
  }
  return state != VectorUnit::VPUState::idle;
}

void VecUnitState::init() {
  auto *sj = (VecUnitJob *) j;
  if (sj->phases.empty()) {
    std::cerr << "Empty job delivered to VPU?" << std::endl;
    throw std::exception();
  }
  LOG_TO_WAVEFORM(STAT_ID(JOB_IDX, vcd_idx), j->job_idx);

  VectorUnit::VPUState first_state;
  int first_phase_read;
  int first_phase_cycles;
  auto front = sj->phases.front();
  if (sj->is_prebuffered) {
    first_phase_read = 0;
    if (front.first == VPUPhase::BROADCAST) {
      first_state = VectorUnit::VPUState::buffered_par;
    } else {
      first_state = VectorUnit::VPUState::buffered_lin;
    }
  } else {
    first_phase_read = sj->linearized_dimension * sj->parallel_dimension * batch_size * data_type_width;
    if (front.first == VPUPhase::BROADCAST) {
      first_state = VectorUnit::VPUState::unbuffered_par;
    } else {
      first_state = VectorUnit::VPUState::unbuffered_lin;
    }
  }
  if (front.first == VPUPhase::BROADCAST) {
    first_phase_cycles = div_ru(sj->linearized_dimension * sj->parallel_dimension * front.second, sz);
  } else {
    first_phase_cycles = sj->linearized_dimension * std::max(batch_size, front.second) * div_ru(sj->parallel_dimension, sz);
  }
  state_transfer(first_state,
                 first_phase_read,
                 0,
                 first_phase_cycles);
  sj->phases.pop();
  loop_cols_tiles = 1;
  loop_row_tiles = 1;
  row_i = 1;
  col_i = 1;
}

VecUnitState::VecUnitState(int sz) : State(2), sz(sz) {
  beats_per_wb = std::max((sz * batch_size) / bytes_per_tx, 1);
}


std::string VecUnitJob::get_job_dims_string() const {
  return std::to_string(parallel_dimension) + " x " + std::to_string(linearized_dimension);
}
VecUnitJob::VecUnitJob(int linearizedDimension,
                       int parallelDimension,
                       bool is_prebuffered,
                       const std::queue<std::pair<VPUPhase, int>> &phases)
    : Job(linearizedDimension * parallelDimension * data_type_width * batch_size),
      linearized_dimension(linearizedDimension),
      parallel_dimension(parallelDimension),
      is_prebuffered(is_prebuffered),
      phases(phases) {}

VecUnitJob::VecUnitJob(int linearizedDimension,
                       int parallelDimension,
                       bool is_prebuffered,
                       const std::vector<std::pair<VPUPhase, int>> &vphases)
    : Job(linearizedDimension * parallelDimension * data_type_width * batch_size),
      linearized_dimension(linearizedDimension),
      parallel_dimension(parallelDimension),
      is_prebuffered(is_prebuffered) {
  for (const auto &q: vphases) {
    phases.push(q);
  }
}

int VecUnitJob::get_type() const {
  return VECTOR_UNIT_IDX;
}
