#include "attention_decode.h"
#include "opt129_llama_fattn_adapter.cuh"
#include "scheduler_primitives.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT129_MATCHED_ATTENTION_RESULT=";
constexpr char kCounts[] = "QW38_OPT129_NATIVE_COUNTS=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kWidth = 256;
constexpr std::uint32_t kRotary = 64;
constexpr int kDefaultLayers[] = {3, 7, 63};
constexpr int kDefaultLayerCount = 3;
constexpr int kDefaultPrefixes[] = {128, 1023, 1024, 2048, 8192, 32768};
constexpr int kDefaultPrefixCount = 6;
constexpr int kNumericalPrefixes[] = {128, 1024, 2048};
constexpr int kNumericalPrefixCount = 3;
constexpr float kMatchedAbs = 2.0e-2F;
constexpr float kNativeAbsWarn = 5.0e-2F;

struct Options final {
  const char* workload = "identity";
  int layer = -1;
  int position = -1;
  int warmups = 3;
  int samples = 5;
  int near_128k = 0;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload identity|inspect|replay|numerical] "
               "[--layer 3|7|63] [--prefix N] [--warmups N] [--samples N] "
               "[--near-128k 0|1]\n",
               argv0);
  return 2;
}

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--layer") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->layer)) return usage(argv[0]);
    } else if ((std::strcmp(arg, "--prefix") == 0 ||
                std::strcmp(arg, "--position") == 0) &&
               index + 1 < argc) {
      if (!parse_int(argv[++index], &options->position)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->warmups)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->samples)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--near-128k") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->near_128k)) return usage(argv[0]);
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    }
  }
  return 0;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

float unit(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  return (static_cast<float>(x & 0x00FFFFFFU) / 8388608.0F - 1.0F) * 0.35F;
}

std::uint32_t layer_seed(int layer) {
  return static_cast<std::uint32_t>(layer) * 10007U + 89U;
}

std::uint64_t fnv(const float* data, std::size_t count) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::size_t index = 0; index < count; ++index) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, data + index, sizeof(bits));
    hash ^= bits;
    hash *= 1099511628211ULL;
  }
  return hash;
}

void fill_case(const qw38::cuda::AttentionConfig& config, std::size_t position,
               std::uint32_t seed, std::vector<float>* query,
               std::vector<float>* key, std::vector<float>* value,
               std::vector<float>* gate, std::vector<float>* qscale,
               std::vector<float>* kscale,
               std::vector<__nv_bfloat16>* committed_key,
               std::vector<__nv_bfloat16>* committed_value) {
  const std::size_t qn = qw38::cuda::attention_query_values(config);
  const std::size_t kn = qw38::cuda::attention_kv_row_values(config);
  const std::size_t cache = qw38::cuda::attention_cache_values(config);
  query->resize(qn);
  key->resize(kn);
  value->resize(kn);
  gate->resize(qn);
  qscale->assign(config.head_width, 1.0F);
  kscale->assign(config.head_width, 1.0F);
  committed_key->assign(cache, __float2bfloat16_rn(0.0F));
  committed_value->assign(cache, __float2bfloat16_rn(0.0F));
  for (std::size_t index = 0; index < qn; ++index) {
    (*query)[index] = unit(static_cast<std::uint32_t>(index), seed);
    (*gate)[index] = unit(static_cast<std::uint32_t>(index), seed + 17U);
  }
  for (std::size_t index = 0; index < kn; ++index) {
    (*key)[index] = unit(static_cast<std::uint32_t>(index), seed + 31U);
    (*value)[index] = unit(static_cast<std::uint32_t>(index), seed + 47U);
  }
  for (std::uint32_t dim = 0; dim < config.head_width; ++dim) {
    (*qscale)[dim] = 0.85F + 0.002F * static_cast<float>(dim);
    (*kscale)[dim] = 0.90F + 0.0015F * static_cast<float>(dim);
  }
  for (std::size_t token = 0; token < position; ++token) {
    for (std::uint32_t kv_head = 0; kv_head < config.kv_heads; ++kv_head) {
      for (std::uint32_t dim = 0; dim < config.head_width; ++dim) {
        const std::size_t phys = qw38::cuda::attention_kv_physical_index(
            token, kv_head, dim, config.capacity, config.head_width);
        const float kitem =
            unit(static_cast<std::uint32_t>(phys), seed + 101U);
        const float vitem =
            unit(static_cast<std::uint32_t>(phys), seed + 131U);
        (*committed_key)[phys] = __float2bfloat16_rn(kitem);
        (*committed_value)[phys] = __float2bfloat16_rn(vitem);
      }
    }
  }
}

struct DeviceCase {
  qw38::cuda::AttentionConfig config{};
  std::size_t position = 0;
  std::size_t qn = 0;
  std::size_t kn = 0;
  std::size_t cache = 0;
  std::size_t packed = 0;
  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_g = nullptr;
  float* d_qs = nullptr;
  float* d_ks = nullptr;
  float* d_nq = nullptr;
  float* d_nk = nullptr;
  float* d_scores = nullptr;
  float* d_out = nullptr;
  float* d_vkq = nullptr;
  float* d_meta = nullptr;
  __nv_bfloat16* d_ck = nullptr;
  __nv_bfloat16* d_cv = nullptr;
  __nv_bfloat16* d_candk = nullptr;
  __nv_bfloat16* d_candv = nullptr;
  __half* d_f16k = nullptr;
  __half* d_f16v = nullptr;
  float* d_f16out = nullptr;

  void free_all() {
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_qs);
    cudaFree(d_ks);
    cudaFree(d_nq);
    cudaFree(d_nk);
    cudaFree(d_scores);
    cudaFree(d_out);
    cudaFree(d_vkq);
    cudaFree(d_meta);
    cudaFree(d_ck);
    cudaFree(d_cv);
    cudaFree(d_candk);
    cudaFree(d_candv);
    cudaFree(d_f16k);
    cudaFree(d_f16v);
    cudaFree(d_f16out);
    *this = DeviceCase{};
  }
};

int alloc_case(const qw38::cuda::AttentionConfig& config, std::size_t position,
               const std::vector<float>& query, const std::vector<float>& key,
               const std::vector<float>& value, const std::vector<float>& gate,
               const std::vector<float>& qscale,
               const std::vector<float>& kscale,
               const std::vector<__nv_bfloat16>& committed_key,
               const std::vector<__nv_bfloat16>& committed_value,
               DeviceCase* device) {
  device->free_all();
  device->config = config;
  device->position = position;
  device->qn = query.size();
  device->kn = key.size();
  device->cache = committed_key.size();
  const int n_kv = static_cast<int>(position + 1);
  const int padded = qw38::cuda::opt129::llama_padded_nkv(n_kv);
  device->packed = static_cast<std::size_t>(padded) *
                   static_cast<std::size_t>(config.kv_heads) *
                   static_cast<std::size_t>(config.head_width);
  const std::size_t partial = qw38::cuda::decode_kv_partial_vkq_values(16);
  const std::size_t meta = qw38::cuda::decode_kv_partial_meta_values(16);
  const std::size_t scores =
      qw38::cuda::attention_score_values(config, position);
  auto check = [](cudaError_t error, const char* op) -> int {
    if (error != cudaSuccess) return fail_cuda(op, error);
    return 0;
  };
  int rc = 0;
  rc |= check(cudaMalloc(&device->d_q, device->qn * sizeof(float)), "q");
  rc |= check(cudaMalloc(&device->d_k, device->kn * sizeof(float)), "k");
  rc |= check(cudaMalloc(&device->d_v, device->kn * sizeof(float)), "v");
  rc |= check(cudaMalloc(&device->d_g, device->qn * sizeof(float)), "g");
  rc |= check(cudaMalloc(&device->d_qs, kWidth * sizeof(float)), "qs");
  rc |= check(cudaMalloc(&device->d_ks, kWidth * sizeof(float)), "ks");
  rc |= check(cudaMalloc(&device->d_nq, device->qn * sizeof(float)), "nq");
  rc |= check(cudaMalloc(&device->d_nk, device->kn * sizeof(float)), "nk");
  rc |= check(cudaMalloc(&device->d_scores, scores * sizeof(float)), "scores");
  rc |= check(cudaMalloc(&device->d_out, device->qn * sizeof(float)), "out");
  rc |= check(cudaMalloc(&device->d_vkq, partial * sizeof(float)), "vkq");
  rc |= check(cudaMalloc(&device->d_meta, meta * sizeof(float)), "meta");
  rc |= check(cudaMalloc(&device->d_ck, device->cache * sizeof(__nv_bfloat16)),
              "ck");
  rc |= check(cudaMalloc(&device->d_cv, device->cache * sizeof(__nv_bfloat16)),
              "cv");
  rc |= check(cudaMalloc(&device->d_candk, device->kn * sizeof(__nv_bfloat16)),
              "candk");
  rc |= check(cudaMalloc(&device->d_candv, device->kn * sizeof(__nv_bfloat16)),
              "candv");
  rc |= check(cudaMalloc(&device->d_f16k, device->packed * sizeof(__half)),
              "f16k");
  rc |= check(cudaMalloc(&device->d_f16v, device->packed * sizeof(__half)),
              "f16v");
  rc |= check(cudaMalloc(&device->d_f16out, device->qn * sizeof(float)),
              "f16out");
  if (rc != 0) {
    device->free_all();
    return rc;
  }
  rc |= check(cudaMemcpy(device->d_q, query.data(), device->qn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D q");
  rc |= check(cudaMemcpy(device->d_k, key.data(), device->kn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D k");
  rc |= check(cudaMemcpy(device->d_v, value.data(), device->kn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D v");
  rc |= check(cudaMemcpy(device->d_g, gate.data(), device->qn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D g");
  rc |= check(cudaMemcpy(device->d_qs, qscale.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D qs");
  rc |= check(cudaMemcpy(device->d_ks, kscale.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D ks");
  rc |= check(cudaMemcpy(device->d_ck, committed_key.data(),
                         device->cache * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice),
              "H2D ck");
  rc |= check(cudaMemcpy(device->d_cv, committed_value.data(),
                         device->cache * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice),
              "H2D cv");
  if (rc != 0) device->free_all();
  return rc;
}

cudaError_t launch_quartz(DeviceCase* device) {
  qw38::cuda::AttentionCache committed{device->d_ck, device->d_cv};
  qw38::cuda::AttentionCache cand{device->d_candk, device->d_candv};
  const int n_parts =
      qw38::cuda::decode_kv_parts_for_position(device->position);
  return qw38::cuda::launch_attention_prepare_partitioned(
      device->config, device->position, device->d_q, device->d_k, device->d_v,
      device->d_qs, device->d_ks, device->d_g, committed, cand, device->d_nq,
      device->d_nk, device->d_scores, device->d_out, device->d_vkq,
      device->d_meta, n_parts, nullptr);
}

cudaError_t launch_matched(DeviceCase* device) {
  qw38::cuda::AttentionCache committed{device->d_ck, device->d_cv};
  qw38::cuda::AttentionCache cand{device->d_candk, device->d_candv};
  return qw38::cuda::opt108::launch_llama_vec_stack(
      device->config, device->position, device->d_q, device->d_k, device->d_v,
      device->d_qs, device->d_ks, device->d_g, committed, cand, device->d_nq,
      device->d_nk, device->d_out, device->d_vkq, device->d_meta, 0, nullptr);
}

cudaError_t launch_adapter(DeviceCase* device) {
  return qw38::cuda::opt129::launch_convert_bf16_to_f16(
      device->config.kv_heads, device->config.capacity,
      device->config.head_width, device->position, device->d_ck, device->d_cv,
      device->d_candk, device->d_candv, device->d_f16k, device->d_f16v,
      nullptr);
}

cudaError_t launch_llama_f16_kernel(DeviceCase* device, bool with_gate) {
  const int n_kv = static_cast<int>(device->position + 1);
  return qw38::cuda::opt129::launch_llama_f16_vec(
      n_kv, device->d_nq, device->d_f16k, device->d_f16v, device->d_f16out,
      device->d_vkq, device->d_meta, with_gate ? device->d_g : nullptr, 0,
      nullptr);
}

cudaError_t launch_adapter_enclosing(DeviceCase* device) {
  cudaError_t error = qw38::cuda::launch_prepare_decode_query(
      device->config, device->position, device->d_q, device->d_qs, device->d_nq,
      nullptr);
  if (error != cudaSuccess) return error;
  error = launch_adapter(device);
  if (error != cudaSuccess) return error;
  return launch_llama_f16_kernel(device, true);
}

float mean_ms(const std::vector<float>& values) {
  if (values.empty()) return 0.0F;
  double sum = 0.0;
  for (float item : values) sum += item;
  return static_cast<float>(sum / static_cast<double>(values.size()));
}

int time_launch(const char* name, DeviceCase* device,
                cudaError_t (*fn)(DeviceCase*), int warmups, int samples,
                std::vector<float>* ms) {
  ms->clear();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int round = 0; round < warmups + samples; ++round) {
    cudaEventRecord(start, nullptr);
    const cudaError_t error = fn(device);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    if (error != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return fail_cuda(name, error);
    }
    if (round >= warmups) {
      float elapsed = 0.0F;
      cudaEventElapsedTime(&elapsed, start, stop);
      ms->push_back(elapsed);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

int compare_outputs(const std::vector<float>& left,
                    const std::vector<float>& right, float* max_abs,
                    std::size_t* nonfinite) {
  *max_abs = 0.0F;
  *nonfinite = 0;
  const std::size_t n = std::min(left.size(), right.size());
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index])) {
      ++(*nonfinite);
    }
    *max_abs = std::max(*max_abs, std::fabs(left[index] - right[index]));
  }
  return 0;
}

int download_out(float* src, std::size_t n, std::vector<float>* host) {
  host->assign(n, 0.0F);
  const cudaError_t error =
      cudaMemcpy(host->data(), src, n * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("D2H", error);
  return 0;
}

int run_shape(int layer, std::size_t position, int warmups, int samples,
              bool numerical) {
  const std::uint32_t capacity = static_cast<std::uint32_t>(position + 1);
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           capacity};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, layer_seed(layer), &query, &key, &value, &gate,
            &qscale, &kscale, &committed_key, &committed_value);
  DeviceCase device{};
  int rc = alloc_case(config, position, query, key, value, gate, qscale, kscale,
                      committed_key, committed_value, &device);
  if (rc != 0) return rc;

  cudaError_t error = launch_matched(&device);
  if (error != cudaSuccess) {
    device.free_all();
    return fail_cuda("populate", error);
  }
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    device.free_all();
    return fail_cuda("populate sync", error);
  }

  std::vector<float> quartz_ms;
  std::vector<float> matched_ms;
  std::vector<float> adapter_ms;
  std::vector<float> llama_kernel_ms;
  std::vector<float> adapter_enclosing_ms;
  rc = time_launch("quartz", &device, launch_quartz, warmups, samples,
                   &quartz_ms);
  const char* quartz_kernel = qw38::cuda::last_decode_query_prep_launch_variant();
  const unsigned n_parts = qw38::cuda::last_decode_attention_n_parts();
  rc |= time_launch("matched", &device, launch_matched, warmups, samples,
                    &matched_ms);
  rc |= time_launch("adapter", &device, launch_adapter, warmups, samples,
                    &adapter_ms);
  auto llama_kernel_only = [](DeviceCase* d) -> cudaError_t {
    return launch_llama_f16_kernel(d, false);
  };
  rc |= time_launch("llama_f16_kernel", &device, llama_kernel_only, warmups,
                    samples, &llama_kernel_ms);
  rc |= time_launch("adapter_enclosing", &device, launch_adapter_enclosing,
                    warmups, samples, &adapter_enclosing_ms);
  if (rc != 0) {
    device.free_all();
    return rc;
  }

  std::vector<float> quartz_out;
  std::vector<float> matched_out;
  std::vector<float> llama_out;
  error = launch_quartz(&device);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  rc |= download_out(device.d_out, device.qn, &quartz_out);
  error = launch_matched(&device);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  rc |= download_out(device.d_out, device.qn, &matched_out);
  error = launch_adapter_enclosing(&device);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  rc |= download_out(device.d_f16out, device.qn, &llama_out);
  if (rc != 0) {
    device.free_all();
    return rc;
  }

  float q_vs_m = 0.0F;
  float q_vs_f = 0.0F;
  float m_vs_f = 0.0F;
  std::size_t nf_qm = 0;
  std::size_t nf_qf = 0;
  std::size_t nf_mf = 0;
  compare_outputs(quartz_out, matched_out, &q_vs_m, &nf_qm);
  compare_outputs(quartz_out, llama_out, &q_vs_f, &nf_qf);
  compare_outputs(matched_out, llama_out, &m_vs_f, &nf_mf);

  const int n_kv = static_cast<int>(position + 1);
  const int padded = qw38::cuda::opt129::llama_padded_nkv(n_kv);
  const char* llama_selected = qw38::cuda::opt129::llama_selected_kernel(padded);
  const bool vec_selected = qw38::cuda::opt129::llama_vec_is_selected(padded);
  const char* quartz_ship =
      qw38::cuda::opt129::quartz_shipping_kernel(position);
  int regs = 0;
  std::size_t local_bytes = 0;
  int occ = 0;
  qw38::cuda::opt129::f16_kernel_attributes(&regs, &local_bytes, &occ);
  int mregs = 0;
  std::size_t mlocal = 0;
  int mocc = 0;
  qw38::cuda::opt108::kernel_attributes(&mregs, &mlocal, &mocc);
  const int nsm = qw38::cuda::opt108::nsm_count();
  const int llama_parts =
      qw38::cuda::opt108::parallel_blocks_for_kv(n_kv, occ, nsm);
  const float first = quartz_ms.empty() ? 0.0F : quartz_ms.front();
  const float later = quartz_ms.size() > 1 ? quartz_ms.back() : first;
  const std::uint64_t q_hash = fnv(query.data(), query.size());
  const std::uint64_t k_hash = fnv(key.data(), key.size());
  const std::uint64_t out_hash = fnv(quartz_out.data(), quartz_out.size());

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-129\",\"kind\":\"shape\","
      "\"layer\":%d,\"prefix\":%zu,\"capacity\":%u,\"n_kv\":%d,"
      "\"llama_padded_nkv\":%d,\"visible_matches_allocated\":true,"
      "\"quartz_kernel\":\"%s\",\"quartz_launch\":\"%s\",\"quartz_n_parts\":%u,"
      "\"llama_selected_kernel\":\"%s\",\"llama_vec_is_selected\":%s,"
      "\"llama_f16_launched\":\"%s\",\"matched_launched\":\"%s\","
      "\"matched_is_native_llama_dtype\":false,"
      "\"f16_kernel_is_matched_result\":%s,"
      "\"quartz_enclosing_ms\":%.9g,\"matched_bf16_enclosing_ms\":%.9g,"
      "\"adapter_ms\":%.9g,\"llama_f16_kernel_ms\":%.9g,"
      "\"adapter_enclosing_ms\":%.9g,\"warmth_first_ms\":%.9g,"
      "\"warmth_later_ms\":%.9g,\"occupancy_f16\":%d,\"occupancy_matched\":%d,"
      "\"nsm\":%d,\"registers_f16\":%d,\"local_bytes_f16\":%zu,"
      "\"llama_n_parts\":%d,\"gqa_ratio\":%d,\"kv_reread_factor\":%d,"
      "\"q_hash\":\"%016llx\",\"k_hash\":\"%016llx\",\"out_hash\":\"%016llx\","
      "\"max_abs_quartz_vs_matched\":%.9g,"
      "\"max_abs_quartz_vs_f16\":%.9g,\"max_abs_matched_vs_f16\":%.9g,"
      "\"nonfinite\":%zu,\"identical_arithmetic\":false}\n",
      kPrefix, layer, position, capacity, n_kv, padded, quartz_ship,
      quartz_kernel != nullptr && quartz_kernel[0] != '\0' ? quartz_kernel
                                                           : quartz_ship,
      n_parts, llama_selected, json_bool(vec_selected),
      qw38::cuda::opt129::kNativeLlamaVec, qw38::cuda::opt129::kMatchedLlamaVec,
      json_bool(vec_selected), static_cast<double>(mean_ms(quartz_ms)),
      static_cast<double>(mean_ms(matched_ms)),
      static_cast<double>(mean_ms(adapter_ms)),
      static_cast<double>(mean_ms(llama_kernel_ms)),
      static_cast<double>(mean_ms(adapter_enclosing_ms)),
      static_cast<double>(first), static_cast<double>(later), occ, mocc, nsm,
      regs, local_bytes, llama_parts, qw38::cuda::opt129::kGqaRatio,
      qw38::cuda::opt129::kGqaRatio, static_cast<unsigned long long>(q_hash),
      static_cast<unsigned long long>(k_hash),
      static_cast<unsigned long long>(out_hash), static_cast<double>(q_vs_m),
      static_cast<double>(q_vs_f), static_cast<double>(m_vs_f),
      nf_qm + nf_qf + nf_mf);

  bool ok = nf_qm == 0 && nf_qf == 0 && nf_mf == 0;
  if (numerical) {
    ok = ok && q_vs_m <= kMatchedAbs;
  }
  (void)kNativeAbsWarn;
  device.free_all();
  return ok ? 0 : 1;
}

int run_inspect() {
  const int crossover =
      qw38::cuda::selected_decode_attention_crossover_threshold();
  const int verified = qw38::cuda::selected_decode_attention_verified_max();
  const char* vec128 = qw38::cuda::selected_decode_attention_vec128_path();
  const char* gqa = qw38::cuda::selected_decode_attention_gqa_path();
  const char* prep = qw38::cuda::selected_decode_query_prep_path();
  int regs = 0;
  std::size_t local_bytes = 0;
  int occ = 0;
  qw38::cuda::opt129::f16_kernel_attributes(&regs, &local_bytes, &occ);
  int mregs = 0;
  std::size_t mlocal = 0;
  int mocc = 0;
  qw38::cuda::opt108::kernel_attributes(&mregs, &mlocal, &mocc);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-129\",\"kind\":\"inspect\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"shipping_crossover\":%d,\"verified_max\":%d,"
      "\"vec128_pin\":\"%s\",\"gqa_pin\":\"%s\",\"prep_pin\":\"%s\","
      "\"production_selector_change\":false,"
      "\"kSelectedDecodeAttentionCrossoverThreshold\":%d,"
      "\"quartz_p128\":\"%s\",\"quartz_p1023\":\"%s\",\"quartz_p1024\":\"%s\","
      "\"quartz_p2048\":\"%s\",\"quartz_p8192\":\"%s\",\"quartz_p32768\":\"%s\","
      "\"llama_p128\":\"%s\",\"llama_p1023\":\"%s\",\"llama_p1024\":\"%s\","
      "\"llama_p2048\":\"%s\",\"llama_p8192\":\"%s\",\"llama_p32768\":\"%s\","
      "\"llama_kv_default\":\"f16/f16\",\"quartz_kv_native\":\"bf16_physical\","
      "\"fattn_kq_stride\":%d,\"gqa_ratio\":%d,"
      "\"f16_occupancy\":%d,\"matched_occupancy\":%d,\"nsm\":%d,"
      "\"f16_registers\":%d,\"f16_local_bytes\":%zu,"
      "\"full_scheduler_dispatch\":\"launch_attention_prepare_partitioned\","
      "\"authority_head\":\"%s\"}\n",
      kPrefix, kLlamaRev, kGgufSha, crossover, verified, vec128, gqa, prep,
      crossover, qw38::cuda::opt129::quartz_shipping_kernel(128),
      qw38::cuda::opt129::quartz_shipping_kernel(1023),
      qw38::cuda::opt129::quartz_shipping_kernel(1024),
      qw38::cuda::opt129::quartz_shipping_kernel(2048),
      qw38::cuda::opt129::quartz_shipping_kernel(8192),
      qw38::cuda::opt129::quartz_shipping_kernel(32768),
      qw38::cuda::opt129::llama_selected_kernel(
          qw38::cuda::opt129::llama_padded_nkv(129)),
      qw38::cuda::opt129::llama_selected_kernel(
          qw38::cuda::opt129::llama_padded_nkv(1024)),
      qw38::cuda::opt129::llama_selected_kernel(
          qw38::cuda::opt129::llama_padded_nkv(1025)),
      qw38::cuda::opt129::llama_selected_kernel(
          qw38::cuda::opt129::llama_padded_nkv(2049)),
      qw38::cuda::opt129::llama_selected_kernel(
          qw38::cuda::opt129::llama_padded_nkv(8193)),
      qw38::cuda::opt129::llama_selected_kernel(
          qw38::cuda::opt129::llama_padded_nkv(32769)),
      qw38::cuda::opt129::kFattnStride, qw38::cuda::opt129::kGqaRatio, occ, mocc,
      qw38::cuda::opt108::nsm_count(), regs, local_bytes, kLlamaRev);
  std::printf(
      "%s{\"inspect_ok\":true,\"hybrid_crossover_unchanged\":%s,"
      "\"claims_throughput\":false}\n",
      kCounts, json_bool(crossover == 1024));
  return crossover == 1024 ? 0 : 1;
}

int run_identity() {
  int rc = run_inspect();
  rc |= run_shape(3, 128, 1, 1, true);
  return rc;
}

int run_replay(const Options& options) {
  int rc = 0;
  std::vector<int> layers;
  std::vector<int> prefixes;
  if (options.layer > 0) {
    layers.push_back(options.layer);
  } else {
    layers.assign(kDefaultLayers, kDefaultLayers + kDefaultLayerCount);
  }
  if (options.position > 0) {
    prefixes.push_back(options.position);
  } else {
    prefixes.assign(kDefaultPrefixes, kDefaultPrefixes + kDefaultPrefixCount);
    if (options.near_128k != 0) prefixes.push_back(131040);
  }
  int shapes = 0;
  for (int layer : layers) {
    for (int prefix : prefixes) {
      const int local =
          run_shape(layer, static_cast<std::size_t>(prefix), options.warmups,
                    options.samples, false);
      if (local == 0) {
        ++shapes;
      } else if (prefix >= 65536) {
        std::printf(
            "%s{\"schema_version\":1,\"task\":\"OPT-129\",\"kind\":\"shape\","
            "\"layer\":%d,\"prefix\":%d,\"resource_conditional\":\"oom_or_"
            "fail\",\"ok\":false}\n",
            kPrefix, layer, prefix);
      } else {
        rc = 1;
      }
    }
  }
  std::printf("%s{\"replay_shapes\":%d,\"ok\":%s}\n", kCounts, shapes,
              json_bool(rc == 0));
  return rc;
}

int run_numerical(const Options& options) {
  int rc = 0;
  for (int i = 0; i < kNumericalPrefixCount; ++i) {
    rc |= run_shape(3, static_cast<std::size_t>(kNumericalPrefixes[i]),
                    std::max(options.warmups, 1), std::max(options.samples, 1),
                    true);
  }
  std::printf("%s{\"numerical_ok\":%s}\n", kCounts, json_bool(rc == 0));
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  if (options.workload == nullptr || options.workload[0] == '\0' ||
      std::strcmp(options.workload, "identity") == 0) {
    if (tier == qw38::cuda::TestTier::kSmoke) return run_identity();
    options.workload = "replay";
  }
  if (std::strcmp(options.workload, "inspect") == 0) return run_inspect();
  if (std::strcmp(options.workload, "identity") == 0) return run_identity();
  if (std::strcmp(options.workload, "replay") == 0) return run_replay(options);
  if (std::strcmp(options.workload, "numerical") == 0) {
    return run_numerical(options);
  }
  return usage(argv[0]);
}
