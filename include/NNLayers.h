/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_NNLAYERS_H
#define PROSE_COMPILER_NNLAYERS_H

#include <vector>

#include "Job.h"
#include "config.h"


void connectJobLists(JobList& source, JobList& target);

void printJobQueue(const std::vector<Job *> &job_queue);

void connectJobs(JobPair& parent, JobPair& child);


#endif // PROSE_COMPILER_NNLAYERS_H
