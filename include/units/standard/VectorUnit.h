/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_VECTORUNIT_H
#define PERF_MODEL_VECTORUNIT_H
#include "State.h"

#include <cstdint>
#include "frontends/standard/StandardUnits.h"
#include <queue>

namespace VectorUnit {
  enum VPUState {
    idle = 0,
    unbuffered_lin,
    unbuffered_par,
    buffered_lin,
    buffered_par,
    write
  };
  enum VPUPhase {
    REDUCE,
    BROADCAST
  };

  class VecUnitState : public State {
public:
    explicit VecUnitState(int sz);

    void init() override;
    int sz;
    VectorUnit::VPUState state = VectorUnit::idle;

    bool increment(const std::function<void(Job *)> &enqueue_job,
                   int &total_idle,
                   int *n_idle_units) override;

    void set_state(int st) override {
      state = (VectorUnit::VPUState) st;
    }
    int get_state() override {
      return (int) state;
    }

    int get_ty_idx() override {
      return VECTOR_UNIT_IDX;
    }

    std::string get_ty_string() override {
      return VECTOR_UNIT_STRING;
    }

private:
    uint8_t idx;
    int beats_per_wb;
  };


  struct VecUnitJob : public Job {
    int linearized_dimension;
    int parallel_dimension;
    std::queue<std::pair<VPUPhase, int>> phases;
    bool is_prebuffered;
    int op_latency = 1;

    [[nodiscard]] std::string get_job_dims_string() const override;
    VecUnitJob(int linearizedDimension, int parallelDimension, bool is_prebuffered, const std::queue<std::pair<VPUPhase, int>> &phases);
    VecUnitJob(int linearizedDimension, int parallelDimension, bool is_prebuffered, const std::vector<std::pair<VPUPhase, int>> &phases);

    int get_type() const override;
  };

};// namespace VectorUnit


#endif//PERF_MODEL_VECTORUNIT_H
