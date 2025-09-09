/*
 * COCOSSim: A Cycle-Accurate Neural Network Accelerator Simulator
 * 
 * Copyright (c) 2025 APEX Lab, Duke University
 * 
 * This software is distributed under the terms of the Apache License 2.0.
 * See LICENSE file for details.
 */

#include "NNLayers.h"
#include "frontends/standard/StandardLayer.h"
#include "global.h"
#include "units/standard/SysArray.h"
#include "units/standard/VectorUnit.h"
#include <frontends/standard/StandardArch.h>

#include <functional>
#include <stdexcept>
#include <cmath>

using namespace frontend::standard;

int force_split = -1;
JobList createSAJobs(int m, int k, int n, int num_jobs) {//M,K,N size, input_dim, output_dim
  JobList jobs;
  for (int i = 0; i < num_jobs; ++i) {
    jobs.push_back(new SystolicArray::SysArrayJob(m, k, n));
  }
  return jobs;
}

static const std::vector<std::pair<VectorUnit::VPUPhase, int>> softmax_phases =
    {{VectorUnit::VPUPhase::BROADCAST, 1}, {VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::BROADCAST, 1}};

using JobCreate_f = std::function<JobPair(const ArchConfig &, const LayerConfig &)>;

JobPair Matmul(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "MM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    JobList jl;
    int required_buff_sz = (M * N + M * std::min(K, a_config.sa_sz_allo)) * batch_size * data_type_width;
    bool output_is_bufferable = required_buff_sz <= buffer_size_bytes;
    if (output_is_bufferable && force_split == -1) {
      jl.push_back(new SystolicArray::SysArrayJob(
          M,
          K,
          N));
      printf("Is fully bufferable, force split (%d)\n", force_split);
    } else {
      printf("Is not fully bufferable or has forced split (%d)\n", force_split);
      float new_N_f = (float) buffer_size_bytes / float(data_type_width * M * batch_size) - (float) a_config.sa_sz_allo;
      int N_max_bufferable = (int) std::floor(new_N_f);
      int splitability = div_ru(N, N_max_bufferable);
      int split_by;
      if (force_split != -1) {
        split_by = force_split;
      } else if (splitability >= n_mxus / 2) {
        split_by = n_mxus;
      } else {
        split_by = splitability;
      }
      int N_divided = N / split_by;
      std::cout << "Split job to " << split_by << " MXUs, split " << splitability << ", nf: " << N_max_bufferable << std::endl;
      for (int i = 0; i * N_max_bufferable < N; i += split_by) {
        jl.push_back(new SystolicArray::SysArrayJob(
            M,
            K,
            std::min(N_divided - (i / split_by) * N_max_bufferable, N_max_bufferable)));
      }
    }
    return {jl, jl};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    std::cout << "Num Job is: " << num_jobs << std::endl;
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs);
    return {matmul_layers, matmul_layers};
  }
}


JobPair Conv(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "MM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    JobList jl;
    int required_buff_sz = (M * N + M * std::min(K, a_config.sa_sz_allo)) * batch_size * data_type_width;
    bool output_is_bufferable = required_buff_sz <= buffer_size_bytes;
    if (output_is_bufferable && force_split == -1) {
      int split = 1;
      if (K>2048){
        split = 4; //split to 4 mxus
      }
      jl.push_back(new SystolicArray::SysArrayJob(
          M,
          K/split,
          N));
    } else {
      printf("Is not fully bufferable or has forced split (%d)\n", force_split);
      float new_N_f = (float) buffer_size_bytes / float(data_type_width * M * batch_size) - (float) a_config.sa_sz_allo;
      int N_max_bufferable = (int) std::floor(new_N_f);
      int splitability = div_ru(N, N_max_bufferable);
      int split_by;
      if (force_split != -1) {
        split_by = force_split;
      } else if (splitability >= n_mxus / 2) {
        split_by = n_mxus;
      } else {
        split_by = splitability;
      }
      int N_divided = N / split_by;
      std::cout << "Split job to " << split_by << " MXUs, split " << splitability << ", nf: " << N_max_bufferable << std::endl;
      for (int i = 0; i * N_max_bufferable < N; i += split_by) {
        jl.push_back(new SystolicArray::SysArrayJob(
            M,
            K,
            std::min(N_divided - (i / split_by) * N_max_bufferable, N_max_bufferable)));
      }
    }
    return {jl, jl};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    std::cout << "Num Job is: " << num_jobs << std::endl;
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs);
    return {matmul_layers, matmul_layers};
  }
}


JobPair MatmulAct(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "MA Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    int num_jobs = std::max(1, int(std::ceil(float(K) / a_config.sa_sz_allo)));

    JobList matmul_layers = createSAJobs(M,
                                         a_config.sa_sz_allo,
                                         N, num_jobs);
    JobList act_layer = {new VectorUnit::VecUnitJob(1, M * K, true, {{VectorUnit::VPUPhase::BROADCAST, 1}})};

    connectJobLists(matmul_layers, act_layer);

    return {matmul_layers, act_layer};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs);
    JobList act_layer = {new VectorUnit::VecUnitJob(1, M * K, true, {{VectorUnit::VPUPhase::BROADCAST, 1}})};

    connectJobLists(matmul_layers, act_layer);

    return {matmul_layers, act_layer};
  }
}

JobPair ActMatmul(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3] * l_config.dimensions[0];
  } else {
    std::cerr << "AM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  JobList act_layer = {new VectorUnit::VecUnitJob(1, M * K, true, {{VectorUnit::VPUPhase::BROADCAST, 1}})};

  if (a_config.ws) {
    int num_jobs = std::max(1, K / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(M,
                                         a_config.sa_sz_allo,
                                         N, num_jobs);
    connectJobLists(act_layer, matmul_layers);

    return {act_layer, matmul_layers};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList matmul_layers = createSAJobs(a_config.sa_sz_allo,
                                         K,
                                         N, num_jobs);
    connectJobLists(act_layer, matmul_layers);

    return {act_layer, matmul_layers};
  }
}

JobPair LayerNorm(const ArchConfig &a_config, const LayerConfig &l_config) {
  int lin_dim, par_dim = 1;
  int op_latency = 1;
  switch (l_config.dimensions.size()) {
    case 1:
      lin_dim = l_config.dimensions[0];
      break;
    case 2:
      par_dim = l_config.dimensions[0];
      lin_dim = l_config.dimensions[1];
      break;
    case 3:
      par_dim = l_config.dimensions[0] * l_config.dimensions[1];
      lin_dim = l_config.dimensions[2] / l_config.dimensions[0];
      if (l_config.dimensions[2] % l_config.dimensions[0] != 0) {
        std::cerr << "linear dimension is not divisible by group size in layernorm..." << std::endl;
        throw std::exception();
      }
      break;
    default:
      std::cout << "Unexpected number of dimensions received in LayerNorm" << std::endl;
      throw std::exception();
  }
  JobList jl;
  int par_acc = par_dim;
  int dec_amt = buffer_size_bytes / data_type_width / lin_dim;
  while (par_acc > 0) {
    jl.push_back(new VectorUnit::VecUnitJob(lin_dim, std::min(dec_amt, par_acc), false,
                                            {{VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::REDUCE, 4}, {VectorUnit::VPUPhase::BROADCAST, 1}}));
    par_acc -= dec_amt;
  }
  return {{jl}, {jl}};
}

JobPair Activation(const ArchConfig &a_config, const LayerConfig &l_config) {
  int sz = 1;
  for (const auto &dim: l_config.dimensions) sz *= dim;

  auto job = new VectorUnit::VecUnitJob(1, sz, false, {{VectorUnit::VPUPhase::BROADCAST, 1}});
  return {{job}, {job}};
}

JobPair Softmax(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M;
  int heads = 1;
  if (l_config.dimensions.size() == 1) {
    M = l_config.dimensions[0];
  } else if (l_config.dimensions.size() == 2) {
    heads = l_config.dimensions[0];
    M = l_config.dimensions[1];
  } else {
    std::cerr << "SM Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  int spl = 1;
  int Mp = M * heads;
  if (heads * M * M * data_type_width * batch_size > buffer_size_bytes || Mp > 1024) {
    spl = std::max(div_ru(heads * M * M * data_type_width * batch_size, buffer_size_bytes), div_ru(Mp, 1024));
    if (spl > Mp) {
      std::cerr << "Can't split this enough to fit inside buffer. That's kinda crazy..." << std::endl;
      throw std::exception();
    }
    Mp /= spl;
  }
  std::cout << "Splitting by " << spl << std::endl;

  int n_jobs = div_ru(div_ru(M * heads, Mp), n_vpus);
  JobList softmax_layer;
  for (int i = 0; i < n_jobs; ++i)
    softmax_layer.push_back(new VectorUnit::VecUnitJob(M, Mp, false, softmax_phases));

  return {softmax_layer, softmax_layer};
}

JobPair SelfAttention(const ArchConfig &a_config, const LayerConfig &l_config) {
  int M, K, N;
  int n_heads = 1;
  if (l_config.dimensions.size() == 3) {
    M = l_config.dimensions[0];
    K = l_config.dimensions[1];
    N = l_config.dimensions[2];
  } else if (l_config.dimensions.size() == 4) {
    n_heads = l_config.dimensions[0];
    M = l_config.dimensions[1];
    K = l_config.dimensions[2];
    N = l_config.dimensions[3];
  } else {
    std::cerr << "SA Not expecting " << l_config.dimensions.size() << " dimensions..." << std::endl;
    throw std::exception();
  }

  if (a_config.ws) {
    force_split = N > 320 ? 4 : -1;
    auto K_proj = Matmul(a_config,
                         LayerConfig("Matmul", {M, K, N}));

    auto Q_proj = Matmul(a_config,
                         LayerConfig("Matmul", {M, K, N}));

    auto V_proj = Matmul(a_config,
                         LayerConfig("Matmul", {M, K, N}));
    force_split = 4;
    auto Dot1 = Matmul(a_config,
                       LayerConfig("Matmul", {n_heads, M, N / n_heads, M}));
    auto Dot2 = Matmul(a_config,
                       LayerConfig("Matmul", {n_heads, M, M, N / n_heads}));
    force_split = N > 320 ? 4 : 1;
    auto O_proj = Matmul(a_config,
                         LayerConfig("Matmul", {n_heads, M, N / n_heads, K}));
    force_split = -1;
    auto softmax_layer = Softmax(a_config, LayerConfig("Softmax", {8, M}));

    connectJobs(K_proj, Q_proj);
    connectJobs(Q_proj, V_proj);
    connectJobs(V_proj, Dot1);
    connectJobs(Dot1, softmax_layer);
    connectJobs(softmax_layer, Dot2);
    connectJobs(Dot2, O_proj);
    return {K_proj.first, O_proj.second};
  } else {
    int num_jobs = std::max(1, M / a_config.sa_sz_allo);
    JobList K_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);
    JobList Q_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);
    JobList V_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);
    JobList Dot1 = createSAJobs(a_config.sa_sz_allo,
                                K,// / m_config.n_heads,
                                M, num_jobs);
    JobList Dot2 = createSAJobs(a_config.sa_sz_allo,
                                M,
                                N, num_jobs);/// m_config.n_heads
    JobList O_proj = createSAJobs(a_config.sa_sz_allo,
                                  K,
                                  N, num_jobs);

    JobList softmax_layer = {new VectorUnit::VecUnitJob(M, M, true,
                                                        {{VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::REDUCE, 1}, {VectorUnit::VPUPhase::BROADCAST, 1}})};
    connectJobLists(K_proj, Q_proj);
    connectJobLists(Q_proj, Dot1);
    connectJobLists(Dot1, softmax_layer);
    connectJobLists(softmax_layer, V_proj);
    connectJobLists(V_proj, Dot2);
    connectJobLists(Dot2, O_proj);


    return {K_proj, O_proj};
  }
}

JobPair MultiHeadSelfAttention(const ArchConfig &a_config, const LayerConfig &l_config) {
  JobList all_heads, all_tails;

  int N = ceil(static_cast<double>(n_heads) / a_config.n_cores);
  std::vector<JobPair> head_jobs;
  for (int i = 0; i < N; ++i) {
    head_jobs.push_back(SelfAttention(a_config, l_config));
  }
  for (int i = 0; i < N - 1; ++i) {
    connectJobLists(head_jobs[i].second, head_jobs[i + 1].first);
  }
  return {head_jobs.front().first, head_jobs.back().second};
}


JobCreate_f getLayerLambda(const std::string &layer_type) {
  if (layer_type == "Matmul")
    return Matmul;
  if (layer_type == "Conv")
    return Conv;
  if (layer_type == "MatmulAct")
    return MatmulAct;
  if (layer_type == "Softmax")
    return Softmax;
  if (layer_type == "Activation")
    return Activation;
  if (layer_type == "LayerNorm")
    return LayerNorm;
  if (layer_type == "SelfAttention")
    return SelfAttention;
  if (layer_type == "MultiHeadSelfAttention")
    return MultiHeadSelfAttention;
  throw std::runtime_error("Unknown layer type: " + layer_type);
}

ArchConfig frontend::standard::arch_config;

std::vector<JobPair> StandardLayer::make_layers(const std::vector<LayerConfig> &layer_configs) const {
  std::vector<JobPair> model_heads;
  JobList jp;
  for (int m = 0; m < model_parallelism; ++m) {
    std::vector<JobPair> lists;
    for (const auto &layer_config: layer_configs) {
      auto layer_f = getLayerLambda(layer_config.layer_type);
      lists.push_back(layer_f(arch_config, layer_config));
    }
    std::cout << "list size: " << lists.size() << std::endl;
    for (int i = 1; i < lists.size(); ++i) {
      connectJobLists(lists[i - 1].second, lists[i].first);
      std::cout << "connect " << i << " to " << (i - 1) << std::endl;
    }
    if (m == 0 || do_par) {
      model_heads.push_back(lists[0]);
      jp = lists.back().second;
    } else {
      connectJobLists(jp, lists[0].first);
      jp = lists[0].second;
    }
  }

  return model_heads;
}
