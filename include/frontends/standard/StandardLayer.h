/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_STANDARD_H
#define PERF_MODEL_STANDARD_H

#include "Job.h"

#include "frontends/LayerParser.h"

namespace frontend::standard {
  struct StandardLayer : LayerParser {
    std::vector<JobPair> make_layers(const std::vector<LayerConfig> &l_configs) const override;
  };
}// namespace frontend::standard

#endif//PERF_MODEL_STANDARD_H
