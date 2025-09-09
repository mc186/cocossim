/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "global.h"
#include <cmath>

int total_jobs = 0;
int jobs_finished = 0;
int bytes_per_tx;
std::vector<std::tuple<uint64_t, bool, int, State *>> to_enqueue;
FILE *vcd = nullptr;
uint64_t gcycles = 0;
int alloc_task_idx = 0;
int model_parallelism = 1;
float freq_sa = 1;
float freq_vu = 1;

bool do_par = false;
char const *rand_chars[] = {"a", "b", "c", "d", "e", "f",
                            "g", "h", "i", "j", "k", "l",
                            "m", "n", "o", "p", "q", "r",
                            "s", "t", "u", "v", "w", "x",
                            "y", "z", "A", "B", "C", "D",
                            "E", "F", "G", "H", "I", "J",
                            "K", "L", "M", "N", "O", "P",
                            "Q", "R", "S", "T", "U", "V",
                            "W", "X", "Y", "Z", "0", "1",
                            "2", "3", "4", "5", "6", "7",
                            "aa", "ab", "ac", "ad", "ae",
                            "af", "ag", "ah", "ai", "aj",
                            "ak", "al", "am", "an", "ao",
                            "ap", "aq", "ar", "as", "at",
                            "au", "av", "aw", "ax", "ay"};

int div_ru(int q, int r) {
    return int(std::ceil(float(q) / (float) r));
}
