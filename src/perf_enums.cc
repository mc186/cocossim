/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "perf_enums.h"
#include <algorithm>

std::string int_to_binary(int n, int sz) {
  std::string s;
  for (int i = 0; i < sz; ++i) {
    s += (n & 1) ? "1" : "0";
    n >>= 1;
  }
  std::reverse(s.begin(), s.end());
  return s;
}
