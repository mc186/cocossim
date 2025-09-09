/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "memory.h"
#include "global.h"

using namespace mem;

namespace mem {
  mem_ty *mem_sys;
  dramsim3::Config *dramsim3config;
  std::unordered_map<uint64_t, State *> address_reads_bkwds_lookup;
  std::unordered_map<uint64_t, State *> address_writes_bkwds_lookup;
}

static int q = 0;

struct PrioritySorter {
    bool operator()(const std::tuple<uint64_t, bool, int, State *> &first, const std::tuple<uint64_t, bool, int, State *> &second) const {
        return std::get<2>(first) < std::get<2>(second);
    }
};

bool mem::try_enqueue_tx() {
    for (int i = 0; i < to_enqueue.size() && i < 64; ++i) {
        auto &pair = to_enqueue[i];
        uint64_t addr = std::get<0>(pair);
        bool is_write = std::get<1>(pair);
        int priority = std::get<2>(pair);
        State *state = std::get<3>(pair);
        if (mem_sys->WillAcceptTransaction(addr, is_write)) {
            mem_sys->AddTransaction(addr, is_write);
            if (is_write) {
                address_writes_bkwds_lookup[addr] = state;
            } else {
                address_reads_bkwds_lookup[addr] = state;
            }
            if (i != to_enqueue.size() - 1) {
                std::swap(pair, to_enqueue.back());
            }
            to_enqueue.pop_back();
            return true;
        }
    }
    return false;
}

void mem::setup() {
  dramsim3config = new dramsim3::Config("./dramsim3/configs/HBM2_8Gb_x128.ini", "./");
  mem_sys = new mem_ty(*dramsim3config, "./", [](uint64_t addr) {
        auto it = address_reads_bkwds_lookup.find(addr);
        if (it != address_reads_bkwds_lookup.end()) {
            State *q = it->second;
            address_reads_bkwds_lookup.erase(it);
            q->mem_read_left -= 1;
        } else {
            std::cerr << "Error: Address " << std::hex << addr << " not found in address_reads_bkwds_lookup" << std::endl;
        } }, [](uint64_t addr) {
        auto it = address_writes_bkwds_lookup.find(addr);
        if (it != address_writes_bkwds_lookup.end()) {
            State *q = it->second;
            address_writes_bkwds_lookup.erase(it);
            q->mem_write_left -= 1;
        } else {
            std::cerr << "Error: Address " << addr << " not found in address_writes_bkwds_lookup" << std::endl;
        } });
  bytes_per_tx = dramsim3config->request_size_bytes;
  std::cout << "REQUEST SIZE BYTES " << bytes_per_tx << std::endl;

}
