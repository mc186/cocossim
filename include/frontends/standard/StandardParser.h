/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_STANDARDPARSER_H
#define PERF_MODEL_STANDARDPARSER_H

#include "frontends/ArchParser.h"
#include "frontends/standard/StandardArch.h"

struct StandardParser : ArchParser {
  Arch *make_arch() override;
  StandardParser(int argc, char **argv) : ArchParser(argc, argv) {}
};
#endif//PERF_MODEL_STANDARDPARSER_H
