/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#ifndef PROSE_COMPILER_PERF_ENUMS_H
#define PROSE_COMPILER_PERF_ENUMS_H
#include <string>

static const int WIDTH_STATE = 3;
static const int WIDTH_IDLE_FROM_MEMORY = 1;
static const int WIDTH_JOB_IDX = 8;
static const int WIDTHS[] = {WIDTH_STATE, WIDTH_IDLE_FROM_MEMORY, WIDTH_JOB_IDX};

static const int STAT_STATE = 0;
static const int STAT_IDLE_FROM_MEMORY = 1;
static const int STAT_JOB_IDX = 2;

static const int total_stat_bits = 2; // this helps to choose names

static const char * NAME_STATE = "_STATE";
static const char * NAME_IDLE_FROM_MEMORY = "_IDLE_FROM_MEMORY";
static const char * NAME_JOB_IDX = "_JOB_IDX";



#define STAT_ID(x, vcdIDX) ((vcdIDX << total_stat_bits) | (STAT_ ## x))
#define STAT_NAME(x) NAME_ ## x
#define STAT_WIDTH(x) (WIDTHS[x])
#define STAT_EXTRACT(x) (x & ((1 << total_stat_bits)-1))

#define PHASE_STATE_IDX (-1)

std::string int_to_binary(int n, int sz);


#endif//PROSE_COMPILER_PERF_ENUMS_H
