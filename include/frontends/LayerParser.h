/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef LAYERPARSER_H
#define LAYERPARSER_H
#include "Job.h"
#include <vector>

struct LayerConfig {
  std::string layer_type;
  std::vector<int> dimensions;
  LayerConfig(const std::string &&layerType, const std::vector<int> &dimensions) : layer_type(layerType), dimensions(dimensions) {}
  LayerConfig() = default;
};


struct LayerParser {
  virtual std::vector<JobPair> make_layers(const std::vector<LayerConfig> &layer_configs) const {
    throw std::runtime_error("LayerParser: Not implemented");
  }
};

#endif //LAYERPARSER_H
