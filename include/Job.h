/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_JOB_H
#define PROSE_COMPILER_JOB_H

#include <cstdint>
#include <iostream>
#include <vector>

extern uint64_t alloc_addr;

struct Job {
  [[nodiscard]] virtual int get_type() const = 0;
  bool batched_weights = false;
  uint64_t addr;
  const uint64_t addr_hold;
  int task_idx;
  int job_idx;

  std::vector<Job *> children;

  int rem_deps;
  bool is_done = false;
  Job(uint64_t alloc_size);

  void add_child(Job *j) {
    children.push_back(j);
    j->rem_deps += 1;
  }

  void reset() {
    addr = addr_hold;
    for (auto *child: children) {
      child->reset();
    }
  }

  virtual std::string get_job_dims_string() const = 0;
  void printDetails() const {
    std::cout << "Job Type: " << get_type()
              << ", Dims: " << get_job_dims_string()
              << ", Address: 0x" << std::hex << addr
              << ", Task Index: " << std::dec << task_idx
              << ", Remaining Dependencies: " << rem_deps
              << ", Children Count: " << children.size() << std::endl;
  }
};

using JobList = std::vector<Job *>;
using JobPair = std::pair<JobList, JobList>;

void jobs_to_dot(std::vector<Job *> &jobs, const std::string &fname = "jobs.dot");

#endif//PROSE_COMPILER_JOB_H
