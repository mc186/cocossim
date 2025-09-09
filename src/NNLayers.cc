/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "NNLayers.h"
#include <vector>

void connectJobLists(JobList &source, JobList &target) {
    for (auto *src: source) {
        for (auto *tgt: target) {
            src->add_child(tgt);
        }
    }
}

void printJobQueue(const std::vector<Job *> &job_queue) {
    std::cout << "Job Queue:" << std::endl;
    for (const auto &job: job_queue) {
        std::cout << "Job Type: " << job->get_type() << "\n"
                  << job->get_job_dims_string() << "\n"
                  << std::endl;
    }
}

void connectJobs(JobPair& parent, JobPair& child) {
  connectJobLists(parent.second, child.first);
}