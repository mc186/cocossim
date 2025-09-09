/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "frontends/standard/StandardArch.h"
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"

using namespace frontend::standard;

StandardArch::StandardArch() {
  for (int i = 0; i < arch_config.n_cores; ++i) states.push_back(new SystolicArray::SysArrayState(arch_config.sa_sz_allo, arch_config.ws));
  for (int i = 0; i < arch_config.n_cores; ++i) states.push_back(new VectorUnit::VecUnitState(arch_config.vu_sz_allo));
}