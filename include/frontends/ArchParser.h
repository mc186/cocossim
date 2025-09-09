/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PERF_MODEL_ARCHPARSER_H
#define PERF_MODEL_ARCHPARSER_H
#include "Arch.h"
#include "global.h"
#include <cstring>

extern std::string layer_file;
extern std::string ofile;

struct ArchParser {
  virtual Arch *make_arch() {
    throw std::runtime_error("make_arch() not implemented for the given frontend.");
  }
  void parse_args(const std::vector<std::pair<std::string, int *>> &input_prs, const std::string &help_str = "") const {
    for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "-i") == 0) {
        layer_file = argv[++i];
      } else if (strcmp(argv[i], "-o") == 0) {
        ofile = argv[++i];
      } else if (strcmp(argv[i], "-f") == 0) {
        freq_sa = std::stof(argv[++i]);
      } else if (strcmp(argv[i], "-h") == 0) {
        std::cerr << "Global Options:\n"
                     "-i <file>     layer input file\n"
                     "-o <file>     output statistic file\n"
                     "-f <float>    frequency (GHz)\n";
        if (help_str != "") {
          std::cerr << "Arch Specific Options:\n"
                    << help_str << std::endl;
        }
        exit(0);
      } else {
        bool found = false;
        for (auto &pr: input_prs) {
          if (strcmp(argv[i], pr.first.c_str()) == 0) {
            *pr.second = std::stoi(argv[++i]);
            found = true;
            break;
          }
        }
        if (not found) {
          throw std::runtime_error("Failed to parse passed flag: '" + std::string(argv[i]) + "'\n");
        }
      }
    }
  }

  int argc;
  char **argv;
  ArchParser(int argc, char **argv) : argc(argc), argv(argv) {}
  ArchParser() = delete;
};
#endif//PERF_MODEL_ARCHPARSER_H
